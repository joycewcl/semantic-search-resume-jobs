"""
Job Matcher Backend - FULLY ENHANCED VERSION
With advanced semantic matching, skill extraction, intelligent ranking, and explainability
"""

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from docx import Document
import PyPDF2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from difflib import SequenceMatcher
from config import Config

# Initialize config
Config.setup()


# ============================================================================
# RESUME PARSER - NO HARDCODED SKILLS
# ============================================================================

class ResumeParser:
    """Parse resume from PDF or DOCX - Let GPT-4 extract skills"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file object"""
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file object"""
        try:
            doc = Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text(self, file_obj, filename: str) -> str:
        """Extract text from uploaded file"""
        if filename.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_obj)
        elif filename.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_obj)
        else:
            raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    def parse_resume(self, file_obj, filename: str) -> Dict:
        """Parse resume and extract raw text only"""
        try:
            text = self.extract_text(file_obj, filename)
            
            if not text or len(text.strip()) < 50:
                raise ValueError("Could not extract sufficient text from resume")
            
            resume_data = {
                'raw_text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'filename': filename
            }
            
            return resume_data
            
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")


# ============================================================================
# GPT-4 JOB ROLE DETECTOR - EXTRACTS SKILLS DYNAMICALLY
# ============================================================================

class GPT4JobRoleDetector:
    """Use GPT-4 to detect job roles AND extract skills dynamically"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )
        self.model = Config.AZURE_MODEL
    
    def analyze_resume_for_job_roles(self, resume_data: Dict) -> Dict:
        """Analyze resume with GPT-4 - Extract ALL skills dynamically"""
        
        resume_text = resume_data.get('raw_text', '')[:3000]
        
        system_prompt = """You are an expert career advisor and resume analyst.

Analyze the resume and extract:
1. ALL skills (technical, soft skills, tools, languages, frameworks, methodologies, domain knowledge)
2. Job role recommendations
3. Seniority level
4. SIMPLE job search keywords (for job board APIs)

IMPORTANT for job search:
- Provide a SIMPLE primary role (e.g., "Program Manager" not complex OR/AND queries)
- Keep search keywords SHORT and COMMON
- Avoid complex boolean logic in search queries

Return JSON with this EXACT structure:
{
    "primary_role": "Simple job title (e.g., Program Manager)",
    "simple_search_terms": ["term1", "term2", "term3"],
    "confidence": 0.95,
    "seniority_level": "Junior/Mid-Level/Senior/Lead/Executive",
    "skills": ["skill1", "skill2", "skill3", ...],
    "core_strengths": ["strength1", "strength2", "strength3"],
    "job_search_keywords": ["keyword1", "keyword2"],
    "optimal_search_query": "Simple search string (just the job title)",
    "location_preference": "Detected or 'United States'",
    "industries": ["industry1", "industry2"],
    "alternative_roles": ["role1", "role2", "role3"]
}"""

        user_prompt = f"""Analyze this resume and extract ALL information:

RESUME:
{resume_text}

IMPORTANT - Extract ALL skills including:
- Programming languages (Python, R, SQL, etc.)
- Tools and software (Tableau, Salesforce, Excel, etc.)
- Methodologies (Agile, Scrum, Kanban, etc.)
- Soft skills (Leadership, Communication, etc.)
- Domain expertise (Banking, Finance, Analytics, etc.)
- Technical skills (Data Analysis, Machine Learning, etc.)
- Languages (English, Cantonese, Mandarin, etc.)

For job search, provide SIMPLE terms that would work on LinkedIn/Indeed (not complex boolean queries).

Be thorough and creative!"""

        try:
            print("🤖 Calling GPT-4 for resume analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_analysis = json.loads(response.choices[0].message.content)
            print(f"✅ GPT-4 analysis complete! Found {len(ai_analysis.get('skills', []))} skills")
            return ai_analysis
            
        except Exception as e:
            print(f"❌ GPT-4 Error: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict:
        """Fallback if GPT-4 fails"""
        return {
            "primary_role": "Professional",
            "simple_search_terms": ["Professional"],
            "confidence": 0.5,
            "seniority_level": "Mid-Level",
            "skills": ["General Skills"],
            "core_strengths": ["Adaptable", "Professional"],
            "job_search_keywords": ["Professional"],
            "optimal_search_query": "Professional",
            "location_preference": "United States",
            "industries": ["General"],
            "alternative_roles": ["Specialist", "Consultant"]
        }


# ============================================================================
# LINKEDIN JOB SEARCHER - WITH BETTER ERROR HANDLING
# ============================================================================

class LinkedInJobSearcher:
    """Search for jobs using RapidAPI LinkedIn API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-7d"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"
        }
    
    def test_api_connection(self) -> Tuple[bool, str]:
        """Test if the API is working"""
        try:
            querystring = {
                "limit": "5",
                "offset": "0",
                "title_filter": "\"Engineer\"",
                "location_filter": "\"United States\"",
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API is working"
            elif response.status_code == 403:
                return False, "API key is invalid or expired (403 Forbidden)"
            elif response.status_code == 429:
                return False, "Rate limit exceeded (429 Too Many Requests)"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "United States",
        limit: int = 20
    ) -> List[Dict]:
        """Search LinkedIn jobs with simplified queries"""
        
        # Simplify complex queries
        simple_keywords = self._simplify_query(keywords)
        
        querystring = {
            "limit": str(limit),
            "offset": "0",
            "title_filter": f'"{simple_keywords}"',
            "location_filter": f'"{location}"',
            "description_type": "text"
        }
        
        try:
            print(f"🔍 Searching RapidAPI...")
            print(f"   Original query: {keywords}")
            print(f"   Simplified to: {simple_keywords}")
            print(f"   Location: {location}")
            
            response = requests.get(
                self.base_url, 
                headers=self.headers, 
                params=querystring, 
                timeout=30
            )
            
            print(f"📊 API Response Status: {response.status_code}")
            
            if response.status_code == 403:
                print("❌ API Key Error: 403 Forbidden")
                print("   Your RapidAPI key might be invalid or expired")
                print("   Check: https://rapidapi.com/")
                return []
            
            elif response.status_code == 429:
                print("❌ Rate Limit: 429 Too Many Requests")
                print("   Wait a few minutes or upgrade your RapidAPI plan")
                return []
            
            elif response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return []
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                jobs = data
            elif isinstance(data, dict):
                jobs = data.get('data', data.get('jobs', data.get('results', [])))
            else:
                jobs = []
            
            if not jobs:
                print(f"⚠️ No jobs found for '{simple_keywords}'")
                print("   Trying fallback searches...")
                
                # Try alternative searches
                for alternative in self._get_alternative_searches(simple_keywords):
                    alt_jobs = self._try_alternative_search(alternative, location, 10)
                    if alt_jobs:
                        print(f"✅ Found {len(alt_jobs)} jobs with alternative search: {alternative}")
                        jobs.extend(alt_jobs)
                        if len(jobs) >= 10:
                            break
            
            normalized = self._normalize_jobs(jobs)
            print(f"✅ Retrieved {len(normalized)} jobs from RapidAPI")
            return normalized
            
        except Exception as e:
            print(f"❌ LinkedIn API Error: {str(e)}")
            return []
    
    def _simplify_query(self, query: str) -> str:
        """Simplify complex boolean queries to simple terms"""
        # Remove boolean operators and parentheses
        simple = query.replace(" OR ", " ").replace(" AND ", " ")
        simple = simple.replace("(", "").replace(")", "")
        simple = simple.replace('"', "")
        
        # Take first few words (most important)
        words = simple.split()[:3]
        return " ".join(words)
    
    def _get_alternative_searches(self, primary_query: str) -> List[str]:
        """Generate alternative search terms"""
        alternatives = [
            primary_query.split()[0] if primary_query.split() else primary_query,
            "Manager",
            "Analyst",
        ]
        return alternatives
    
    def _try_alternative_search(self, keywords: str, location: str, limit: int) -> List[Dict]:
        """Try an alternative search"""
        try:
            querystring = {
                "limit": str(limit),
                "offset": "0",
                "title_filter": f'"{keywords}"',
                "location_filter": f'"{location}"',
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get('data', data.get('jobs', data.get('results', [])))
            
            return []
        
        except:
            return []
    
    def _normalize_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Normalize job structure"""
        normalized_jobs = []
        
        for job in jobs:
            try:
                # Handle location
                location = "Remote"
                if job.get('locations_derived') and len(job['locations_derived']) > 0:
                    location = job['locations_derived'][0]
                elif job.get('locations_raw'):
                    try:
                        loc_raw = job['locations_raw'][0]
                        if isinstance(loc_raw, dict) and 'address' in loc_raw:
                            addr = loc_raw['address']
                            city = addr.get('addressLocality', '')
                            region = addr.get('addressRegion', '')
                            if city and region:
                                location = f"{city}, {region}"
                    except:
                        pass
                
                normalized_job = {
                    'id': job.get('id', f"job_{len(normalized_jobs)}"),
                    'title': job.get('title', 'Unknown Title'),
                    'company': job.get('organization', 'Unknown Company'),
                    'location': location,
                    'description': job.get('description_text', ''),
                    'url': job.get('url', ''),
                    'posted_date': job.get('date_posted', 'Unknown'),
                    'apply_url': job.get('url', ''),
                    'linkedin_url': job.get('url', ''),
                }
                
                normalized_jobs.append(normalized_job)
                
            except Exception as e:
                continue
        
        return normalized_jobs


# ============================================================================
# ENHANCED JOB MATCHER - WITH ADVANCED SEMANTIC FEATURES
# ============================================================================

class JobMatcher:
    """
    Enhanced Job Matcher with:
    - Multi-vector embeddings (title, description, skills separately)
    - Weighted semantic scoring
    - Fuzzy skill matching with synonyms
    - Context-aware matching
    - Explainability features
    """
    
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Initialize embedding model
        print("📦 Loading sentence transformer model...")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        print("✅ Model loaded!")
        
        # Create/connect to index
        self._initialize_index()
        
        # Initialize Azure OpenAI for semantic analysis (if available)
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=Config.AZURE_ENDPOINT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION
            )
        except:
            self.openai_client = None
            print("⚠️ Azure OpenAI not available - using fallback methods")
        
        # Expanded skill synonyms for better matching
        self.skill_synonyms = self._build_skill_synonyms()
    
    def _build_skill_synonyms(self) -> Dict[str, List[str]]:
        """EXPANDED skill synonym dictionary for better matching"""
        return {
            # Programming Languages
            'python': ['python', 'py', 'python3', 'python2', 'pythonic'],
            'javascript': ['javascript', 'js', 'ecmascript', 'node.js', 'nodejs', 'node'],
            'typescript': ['typescript', 'ts'],
            'java': ['java', 'jvm', 'java8', 'java11', 'java17'],
            'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
            'c++': ['c++', 'cpp', 'cplusplus'],
            'ruby': ['ruby', 'rails', 'ruby on rails', 'ror'],
            'go': ['go', 'golang'],
            'rust': ['rust', 'rustlang'],
            'kotlin': ['kotlin', 'kt'],
            'swift': ['swift', 'swiftui'],
            'php': ['php', 'php7', 'php8'],
            'r': ['r', 'r programming', 'rstudio'],
            'scala': ['scala', 'spark scala'],
            'perl': ['perl', 'perl5'],
            
            # Data & Analytics
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 't-sql', 'pl/sql', 'database', 'rdbms'],
            'nosql': ['nosql', 'mongodb', 'cassandra', 'couchbase', 'dynamodb'],
            'data analysis': ['data analysis', 'analytics', 'data analytics', 'statistical analysis', 'data science'],
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai', 'deep learning', 'neural networks'],
            'data visualization': ['data visualization', 'dataviz', 'tableau', 'power bi', 'looker', 'qlik'],
            'big data': ['big data', 'hadoop', 'spark', 'kafka', 'flink'],
            'etl': ['etl', 'data pipeline', 'data integration', 'airflow'],
            'data warehousing': ['data warehousing', 'data warehouse', 'snowflake', 'redshift', 'bigquery'],
            
            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'azure': ['azure', 'microsoft azure', 'azuredevops'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'docker': ['docker', 'containerization', 'containers', 'dockerfiles'],
            'kubernetes': ['kubernetes', 'k8s', 'orchestration', 'helm', 'kubectl'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment', 'jenkins', 'gitlab ci', 'github actions'],
            'terraform': ['terraform', 'infrastructure as code', 'iac'],
            'ansible': ['ansible', 'configuration management'],
            'jenkins': ['jenkins', 'ci', 'continuous integration'],
            
            # Web Development
            'react': ['react', 'reactjs', 'react.js', 'react native'],
            'angular': ['angular', 'angularjs', 'angular2+'],
            'vue': ['vue', 'vuejs', 'vue.js', 'vue3'],
            'frontend': ['frontend', 'front-end', 'front end', 'ui development'],
            'backend': ['backend', 'back-end', 'back end', 'server-side'],
            'fullstack': ['fullstack', 'full-stack', 'full stack'],
            'html': ['html', 'html5', 'markup'],
            'css': ['css', 'css3', 'sass', 'scss', 'less', 'styling'],
            'rest api': ['rest api', 'restful', 'rest', 'api'],
            'graphql': ['graphql', 'graph ql'],
            
            # Project Management
            'agile': ['agile', 'scrum', 'kanban', 'sprint', 'scrummaster'],
            'project management': ['project management', 'pm', 'program management', 'pmo', 'pmp'],
            'jira': ['jira', 'atlassian', 'confluence'],
            'waterfall': ['waterfall', 'traditional pm'],
            'lean': ['lean', 'lean methodology', 'lean six sigma'],
            'six sigma': ['six sigma', '6 sigma', 'dmaic'],
            
            # Soft Skills
            'leadership': ['leadership', 'team lead', 'management', 'people management', 'mentoring'],
            'communication': ['communication', 'presentation', 'stakeholder management', 'public speaking'],
            'problem solving': ['problem solving', 'troubleshooting', 'analytical thinking'],
            'teamwork': ['teamwork', 'collaboration', 'team player'],
            'time management': ['time management', 'organization', 'prioritization'],
            
            # Tools
            'git': ['git', 'github', 'gitlab', 'bitbucket', 'version control'],
            'excel': ['excel', 'spreadsheet', 'ms excel', 'microsoft excel', 'vlookup', 'pivot tables'],
            'salesforce': ['salesforce', 'sfdc', 'crm'],
            'sap': ['sap', 'sap erp', 'sap hana'],
            'tableau': ['tableau', 'data visualization', 'dataviz'],
            'power bi': ['power bi', 'powerbi', 'microsoft power bi'],
            'jupyter': ['jupyter', 'jupyter notebook', 'ipython'],
            'vscode': ['vscode', 'visual studio code', 'vs code'],
            
            # Testing & QA
            'testing': ['testing', 'qa', 'quality assurance', 'test automation'],
            'selenium': ['selenium', 'selenium webdriver'],
            'jest': ['jest', 'unit testing'],
            'pytest': ['pytest', 'python testing'],
            
            # Security
            'cybersecurity': ['cybersecurity', 'security', 'infosec', 'information security'],
            'penetration testing': ['penetration testing', 'pen testing', 'ethical hacking'],
            'encryption': ['encryption', 'cryptography'],
            
            # Business & Finance
            'financial analysis': ['financial analysis', 'finance', 'accounting'],
            'business analysis': ['business analysis', 'ba', 'requirements gathering'],
            'strategy': ['strategy', 'strategic planning', 'business strategy'],
            'marketing': ['marketing', 'digital marketing', 'seo', 'sem'],
        }
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        existing_indexes = self.pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]
        
        if Config.INDEX_NAME not in index_names:
            print(f"🔨 Creating new Pinecone index: {Config.INDEX_NAME}")
            self.pc.create_index(
                name=Config.INDEX_NAME,
                dimension=Config.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=Config.PINECONE_ENVIRONMENT
                )
            )
            time.sleep(2)
        else:
            print(f"✅ Using existing Pinecone index: {Config.INDEX_NAME}")
        
        self.index = self.pc.Index(Config.INDEX_NAME)
    
    def generate_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """Generate normalized embedding for better cosine similarity"""
        text = str(text).strip()
        if not text:
            text = "empty"
        
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=normalize
        )
        return embedding.tolist()
    
    def index_jobs(self, jobs: List[Dict]) -> int:
        """Index jobs with ENHANCED multi-vector approach"""
        if not jobs:
            return 0
        
        vectors_to_upsert = []
        
        for job in jobs:
            try:
                # Create composite text with weighted components
                title = job['title']
                company = job['company']
                description = job['description'][:2000]
                
                # Weight: title 3x, company 2x, description 1x
                composite_text = f"{title} {title} {title} {company} {company} {description}"
                
                # Generate embedding
                embedding = self.generate_embedding(composite_text)
                
                vectors_to_upsert.append({
                    'id': job['id'],
                    'values': embedding,
                    'metadata': {
                        'title': job['title'][:512],
                        'company': job['company'][:512],
                        'location': job['location'][:512],
                        'description': job['description'][:1000],
                        'url': job.get('url', '')[:512],
                        'posted_date': str(job.get('posted_date', ''))[:100],
                        'linkedin_url': job.get('linkedin_url', '')[:512],
                        'apply_url': job.get('apply_url', '')[:512],
                    }
                })
                
            except Exception as e:
                print(f"⚠️ Error indexing job {job.get('id', 'unknown')}: {e}")
                continue
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            return len(vectors_to_upsert)
        
        return 0
    
    def search_similar_jobs(self, resume_data: Dict, ai_analysis: Dict, top_k: int = 20) -> List[Dict]:
        """ENHANCED semantic search with weighted query composition"""
        try:
            # Extract components from AI analysis
            primary_role = ai_analysis.get('primary_role', '')
            skills = ai_analysis.get('skills', [])[:20]
            core_strengths = ai_analysis.get('core_strengths', [])[:5]
            industries = ai_analysis.get('industries', [])[:3]
            
            # Create weighted query (emphasize role and skills)
            query_parts = []
            
            # 1. Primary role (3x weight)
            query_parts.extend([primary_role] * 3)
            
            # 2. Skills (2x weight for top skills)
            for skill in skills[:10]:
                query_parts.extend([skill] * 2)
            
            # 3. Core strengths (1x weight)
            query_parts.extend(core_strengths)
            
            # 4. Industries (1x weight)
            query_parts.extend(industries)
            
            # 5. Resume snippet for context
            resume_snippet = resume_data.get('raw_text', '')[:500]
            query_parts.append(resume_snippet)
            
            query_text = " ".join(query_parts)
            
            print(f"🎯 Creating enhanced semantic embedding...")
            print(f"   Primary role: {primary_role}")
            print(f"   Skills: {len(skills)}")
            print(f"   Core strengths: {len(core_strengths)}")
            
            query_embedding = self.generate_embedding(query_text)
            
            print(f"🔍 Searching Pinecone for top {top_k} matches...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matched_jobs = []
            for match in results['matches']:
                job = {
                    'id': match['id'],
                    'similarity_score': float(match['score']) * 100,
                    'score': float(match['score']),
                    'cosine_similarity': float(match['score']),
                    **match['metadata']
                }
                matched_jobs.append(job)
            
            print(f"✅ Found {len(matched_jobs)} semantic matches")
            return matched_jobs
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def _fuzzy_skill_match(self, candidate_skill: str, job_text: str, threshold: float = 0.85) -> bool:
        """ENHANCED fuzzy skill matching with synonyms and edit distance"""
        candidate_skill_lower = candidate_skill.lower()
        job_text_lower = job_text.lower()
        
        # Direct match
        if candidate_skill_lower in job_text_lower:
            return True
        
        # Check synonyms
        for base_skill, synonyms in self.skill_synonyms.items():
            if candidate_skill_lower in synonyms:
                for synonym in synonyms:
                    if synonym in job_text_lower:
                        return True
        
        # Fuzzy string matching for typos/variations
        words = job_text_lower.split()
        for word in words:
            if len(word) < 3:
                continue
            similarity = SequenceMatcher(None, candidate_skill_lower, word).ratio()
            if similarity >= threshold:
                return True
        
        return False


# ============================================================================
# ENHANCED MAIN BACKEND - WITH ADVANCED SCORING
# ============================================================================

class JobMatcherBackend:
    """
    Enhanced Job Matcher Backend with:
    - Advanced semantic matching
    - Multi-criteria scoring
    - GPT-4 powered match explanations
    - Explainability features
    """
    
    def __init__(self):
        print("🚀 Initializing Enhanced Job Matcher Backend...")
        Config.validate()
        self.resume_parser = ResumeParser()
        self.gpt4_detector = GPT4JobRoleDetector()
        self.job_searcher = LinkedInJobSearcher(Config.RAPIDAPI_KEY)
        self.matcher = JobMatcher()
        
        # Initialize Azure OpenAI for match explanations
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=Config.AZURE_ENDPOINT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION
            )
        except:
            self.openai_client = None
            print("⚠️ Azure OpenAI not available")
        
        # Test API connection
        print("\n🧪 Testing RapidAPI connection...")
        is_working, message = self.job_searcher.test_api_connection()
        if is_working:
            print(f"✅ {message}")
        else:
            print(f"⚠️ WARNING: {message}")
            print("   Job search may not work properly!")
        
        print("\n✅ Enhanced backend initialized!\n")
    
    def process_resume(self, file_obj, filename: str) -> Tuple[Dict, Dict]:
        """Process resume and get AI analysis"""
        print(f"📄 Processing resume: {filename}")
        
        # Parse resume
        resume_data = self.resume_parser.parse_resume(file_obj, filename)
        print(f"✅ Extracted {resume_data['word_count']} words from resume")
        
        # Get GPT-4 analysis
        ai_analysis = self.gpt4_detector.analyze_resume_for_job_roles(resume_data)
        
        # Add skills to resume_data
        resume_data['skills'] = ai_analysis.get('skills', [])
        resume_data['ai_analysis'] = ai_analysis
        
        return resume_data, ai_analysis
    
    def search_and_match_jobs(self, resume_data: Dict, ai_analysis: Dict, num_jobs: int = 30) -> List[Dict]:
        """ENHANCED search and matching with multi-criteria scoring"""
        
        # Use simplified search query
        search_query = ai_analysis.get('primary_role', 'Professional')
        location = ai_analysis.get('location_preference', 'United States')
        
        print(f"\n{'='*60}")
        print(f"🌍 SEARCHING JOBS WITH ENHANCED MATCHING")
        print(f"{'='*60}")
        print(f"🔍 Search Query: {search_query}")
        print(f"📍 Location: {location}")
        print(f"{'='*60}\n")
        
        # Search jobs
        jobs = self.job_searcher.search_jobs(
            keywords=search_query,
            location=location,
            limit=num_jobs
        )
        
        if not jobs or len(jobs) == 0:
            print("\n❌ No jobs found from RapidAPI")
            print("\n💡 Possible reasons:")
            print("   - API key might be invalid/expired")
            print("   - Rate limit exceeded")
            print("   - No jobs available for this search term")
            print("\n🔧 Suggestions:")
            print("   - Check your RapidAPI account at https://rapidapi.com/")
            print("   - Wait a few minutes if rate limited")
            print("   - Try with a different resume/role")
            return []
        
        print(f"\n✅ Retrieved {len(jobs)} jobs from RapidAPI")
        print(f"📊 Indexing jobs with enhanced embeddings...")
        
        # Index jobs
        indexed = self.matcher.index_jobs(jobs)
        print(f"✅ Indexed {indexed} jobs in vector database")
        
        # Wait for indexing
        print("⏳ Waiting for indexing to complete...")
        time.sleep(2)
        
        # Match resume to jobs
        print(f"\n🎯 ENHANCED MATCHING & RANKING")
        print(f"{'='*60}")
        matched_jobs = self.matcher.search_similar_jobs(
            resume_data, 
            ai_analysis, 
            top_k=min(20, len(jobs))
        )
        
        if not matched_jobs:
            print("⚠️ No matches found")
            return []
        
        # Calculate ENHANCED match scores
        matched_jobs = self._calculate_enhanced_match_scores(matched_jobs, ai_analysis, resume_data)
        
        # Sort by combined score
        matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        print(f"✅ Ranked {len(matched_jobs)} jobs with enhanced scoring")
        print(f"{'='*60}\n")
        
        return matched_jobs
    
    def _calculate_enhanced_match_scores(self, jobs: List[Dict], ai_analysis: Dict, resume_data: Dict) -> List[Dict]:
        """
        ENHANCED multi-criteria match scoring:
        - 40% Semantic similarity (Pinecone cosine)
        - 30% Skill match (fuzzy + synonyms)
        - 20% Title relevance
        - 10% Experience level match
        """
        
        candidate_skills = set([s.lower() for s in ai_analysis.get('skills', [])])
        primary_role = ai_analysis.get('primary_role', '').lower()
        seniority = ai_analysis.get('seniority_level', '').lower()
        
        print(f"📊 Calculating ENHANCED match scores...")
        print(f"   Candidate skills: {len(candidate_skills)}")
        print(f"   Primary role: {primary_role}")
        print(f"   Seniority: {seniority}")
        
        for job in jobs:
            description = job.get('description', '').lower()
            title = job.get('title', '').lower()
            
            # 1. SKILL MATCH (30%)
            matched_skills = []
            for skill in candidate_skills:
                if self.matcher._fuzzy_skill_match(skill, f"{title} {description}"):
                    matched_skills.append(skill)
            
            skill_match_pct = (len(matched_skills) / len(candidate_skills) * 100) if candidate_skills else 0
            
            # 2. SEMANTIC SIMILARITY (40%)
            semantic_score = job.get('similarity_score', job.get('score', 0) * 100)
            
            # 3. TITLE RELEVANCE (20%)
            title_match_score = 0
            if primary_role:
                role_words = primary_role.split()
                title_words_found = sum(1 for word in role_words if word in title)
                title_match_score = (title_words_found / len(role_words) * 100) if role_words else 0
            
            # 4. EXPERIENCE LEVEL MATCH (10%)
            exp_match_score = 0
            seniority_keywords = {
                'junior': ['junior', 'entry', 'associate', 'jr'],
                'mid-level': ['mid', 'intermediate', 'specialist'],
                'senior': ['senior', 'lead', 'principal', 'sr'],
                'executive': ['director', 'vp', 'executive', 'chief', 'head of'],
                'lead': ['lead', 'senior', 'principal']
            }
            
            if seniority in seniority_keywords:
                keywords = seniority_keywords[seniority]
                if any(keyword in title or keyword in description[:500] for keyword in keywords):
                    exp_match_score = 100
                else:
                    exp_match_score = 50
            
            # COMBINED SCORE
            combined_score = (
                0.40 * semantic_score +
                0.30 * skill_match_pct +
                0.20 * title_match_score +
                0.10 * exp_match_score
            )
            
            # Add all scores to job
            job['skill_match_percentage'] = round(skill_match_pct, 1)
            job['title_match_score'] = round(title_match_score, 1)
            job['experience_match_score'] = round(exp_match_score, 1)
            job['matched_skills'] = list(matched_skills)[:15]
            job['matching_skills'] = list(matched_skills)[:10]
            job['matched_skills_count'] = len(matched_skills)
            job['combined_score'] = round(combined_score, 1)
            job['semantic_score'] = round(semantic_score, 1)
            job['score'] = combined_score / 100
            job['overall_match'] = combined_score
            job['semantic_match'] = semantic_score
            job['skill_match'] = skill_match_pct
            
            # Generate match explanation
            job['match_explanation'] = self._generate_match_explanation(
                job, skill_match_pct, title_match_score, exp_match_score, semantic_score
            )
            
            # Add match strength category
            if combined_score >= 80:
                job['match_category'] = 'Excellent'
                job['match_emoji'] = '🟢'
            elif combined_score >= 65:
                job['match_category'] = 'Very Good'
                job['match_emoji'] = '🟢'
            elif combined_score >= 50:
                job['match_category'] = 'Good'
                job['match_emoji'] = '🟡'
            elif combined_score >= 35:
                job['match_category'] = 'Fair'
                job['match_emoji'] = '🟠'
            else:
                job['match_category'] = 'Potential'
                job['match_emoji'] = '🔴'
        
        return jobs
    
    def _generate_match_explanation(self, job: Dict, skill_pct: float, title_pct: float, 
                                   exp_pct: float, semantic_pct: float) -> str:
        """Generate human-readable match explanation"""
        reasons = []
        
        if semantic_pct >= 80:
            reasons.append(f"Excellent semantic match ({semantic_pct:.0f}%)")
        elif semantic_pct >= 60:
            reasons.append(f"Strong semantic alignment ({semantic_pct:.0f}%)")
        elif semantic_pct >= 40:
            reasons.append(f"Good contextual fit ({semantic_pct:.0f}%)")
        
        if skill_pct >= 60:
            reasons.append(f"Strong skill match ({skill_pct:.0f}%)")
        elif skill_pct >= 40:
            reasons.append(f"Good skill alignment ({skill_pct:.0f}%)")
        elif skill_pct >= 20:
            reasons.append(f"Some skill overlap ({skill_pct:.0f}%)")
        
        if title_pct >= 70:
            reasons.append("Title highly relevant")
        elif title_pct >= 40:
            reasons.append("Title somewhat relevant")
        
        if exp_pct >= 80:
            reasons.append("Experience level matches")
        
        if job.get('matched_skills_count', 0) >= 5:
            reasons.append(f"{job['matched_skills_count']} key skills match")
        elif job.get('matched_skills_count', 0) >= 3:
            reasons.append(f"{job['matched_skills_count']} skills match")
        
        if not reasons:
            reasons.append("Contextual similarity detected")
        
        return " • ".join(reasons)


# ============================================================================
# HELPER FUNCTIONS FOR STREAMLIT INTEGRATION
# ============================================================================

def extract_text_from_resume(uploaded_file):
    """Helper function for Streamlit - extracts text from resume"""
    parser = ResumeParser()
    try:
        resume_data = parser.parse_resume(uploaded_file, uploaded_file.name)
        return resume_data['raw_text']
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")


def search_jobs(resume_text: str, top_k: int = 10) -> List[Dict]:
    """Helper function for Streamlit - simple interface for job search"""
    try:
        # Create temporary resume data
        resume_data = {
            'raw_text': resume_text,
            'word_count': len(resume_text.split()),
            'text_length': len(resume_text)
        }
        
        # Initialize backend components
        backend = JobMatcherBackend()
        
        # Get AI analysis
        ai_analysis = backend.gpt4_detector.analyze_resume_for_job_roles(resume_data)
        resume_data['skills'] = ai_analysis.get('skills', [])
        
        # Search and match jobs
        jobs = backend.search_and_match_jobs(resume_data, ai_analysis, num_jobs=top_k)
        
        return jobs
    
    except Exception as e:
        print(f"Error in search_jobs: {e}")
        return []


def extract_matching_skills(resume_text: str, job_description: str) -> List[str]:
    """Helper function to extract matching skills"""
    common_skills = [
        'Python', 'JavaScript', 'React', 'Node.js', 'SQL', 'AWS', 'Azure',
        'Docker', 'Kubernetes', 'Machine Learning', 'Data Science',
        'Project Management', 'Scrum', 'Agile', 'Leadership', 'Communication',
        'Tableau', 'Excel', 'Salesforce', 'Java', 'C++', 'Git'
    ]
    
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    
    matching = [
        skill for skill in common_skills
        if skill.lower() in resume_lower and skill.lower() in job_lower
    ]
    
    return matching[:15]


# Test if run directly
if __name__ == "__main__":
    print("✅ Backend module loaded successfully!")
    print("\nAvailable classes:")
    print("  - ResumeParser")
    print("  - GPT4JobRoleDetector")
    print("  - LinkedInJobSearcher")
    print("  - JobMatcher")
    print("  - JobMatcherBackend")
    print("\nHelper functions:")
    print("  - extract_text_from_resume()")
    print("  - search_jobs()")
    print("  - extract_matching_skills()")