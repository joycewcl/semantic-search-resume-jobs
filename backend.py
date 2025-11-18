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
import streamlit as st

# ============================================================================
# ENVIRONMENT VARIABLE HELPER FOR STREAMLIT CLOUD
# ============================================================================

def get_env_variable(key: str) -> str:
    """
    Get environment variable from Streamlit secrets (cloud) or .env (local)
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fallback to environment variables (for local .env)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    value = os.getenv(key)
    
    if not value:
        raise ValueError(f"❌ Missing environment variable: {key}")
    
    return value


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
            azure_endpoint=get_env_variable('AZURE_OPENAI_ENDPOINT'),
            api_key=get_env_variable('AZURE_OPENAI_KEY'),
            api_version="2024-02-15-preview"
        )
        self.model = get_env_variable('AZURE_OPENAI_DEPLOYMENT')
    
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
    Enhanced Job Matcher with semantic matching and skill analysis
    """
    
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=get_env_variable('PINECONE_API_KEY'))
        
        # Initialize embedding model
        print("📦 Loading sentence transformer model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded!")
        
        # Create/connect to index
        self._initialize_index()
        
        # Skill synonyms for better matching
        self.skill_synonyms = self._build_skill_synonyms()
    
    def _build_skill_synonyms(self) -> Dict[str, List[str]]:
        """Build skill synonym dictionary for better matching"""
        return {
            # Programming Languages
            'python': ['python', 'py', 'python3', 'pythonic'],
            'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
            'java': ['java', 'jvm'],
            'sql': ['sql', 'mysql', 'postgresql', 'database'],
            
            # Data & Analytics
            'data analysis': ['data analysis', 'analytics', 'data science'],
            'machine learning': ['machine learning', 'ml', 'ai'],
            'tableau': ['tableau', 'data visualization'],
            
            # Cloud
            'aws': ['aws', 'amazon web services'],
            'azure': ['azure', 'microsoft azure'],
            
            # Project Management
            'agile': ['agile', 'scrum', 'kanban'],
            'project management': ['project management', 'pm', 'pmp'],
        }
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        index_name = get_env_variable('PINECONE_INDEX_NAME')
        
        existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"🔨 Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=get_env_variable('PINECONE_ENVIRONMENT')
                )
            )
            time.sleep(2)
        else:
            print(f"✅ Using existing Pinecone index: {index_name}")
        
        self.index = self.pc.Index(index_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        text = str(text).strip()
        if not text:
            text = "empty"
        
        embedding = self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        return embedding.tolist()
    
    def index_jobs(self, jobs: List[Dict]) -> int:
        """Index jobs in Pinecone"""
        if not jobs:
            return 0
        
        vectors_to_upsert = []
        
        for job in jobs:
            try:
                # Create composite text
                title = job['title']
                company = job['company']
                description = job['description'][:2000]
                
                composite_text = f"{title} {title} {title} {company} {company} {description}"
                
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
                print(f"⚠️ Error indexing job: {e}")
                continue
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            return len(vectors_to_upsert)
        
        return 0
    
    def search_similar_jobs(self, resume_data: Dict, ai_analysis: Dict, top_k: int = 20) -> List[Dict]:
        """Search for similar jobs"""
        try:
            # Extract components
            primary_role = ai_analysis.get('primary_role', '')
            skills = ai_analysis.get('skills', [])[:20]
            core_strengths = ai_analysis.get('core_strengths', [])[:5]
            
            # Create query
            query_parts = [primary_role] * 3
            query_parts.extend(skills)
            query_parts.extend(core_strengths)
            
            query_text = " ".join(query_parts)
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
                    **match['metadata']
                }
                matched_jobs.append(job)
            
            return matched_jobs
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def _fuzzy_skill_match(self, candidate_skill: str, job_text: str) -> bool:
        """Fuzzy skill matching"""
        candidate_skill_lower = candidate_skill.lower()
        job_text_lower = job_text.lower()
        
        if candidate_skill_lower in job_text_lower:
            return True
        
        # Check synonyms
        for base_skill, synonyms in self.skill_synonyms.items():
            if candidate_skill_lower in synonyms:
                for synonym in synonyms:
                    if synonym in job_text_lower:
                        return True
        
        return False


# ============================================================================
# MAIN BACKEND
# ============================================================================

class JobMatcherBackend:
    """Main backend class"""
    
    def __init__(self):
        print("🚀 Initializing Job Matcher Backend...")
        Config.validate()
        self.resume_parser = ResumeParser()
        self.gpt4_detector = GPT4JobRoleDetector()
        self.job_searcher = LinkedInJobSearcher(get_env_variable('RAPIDAPI_KEY'))
        self.matcher = JobMatcher()
        print("✅ Backend initialized!\n")
    
    def process_resume(self, file_obj, filename: str) -> Tuple[Dict, Dict]:
        """Process resume"""
        resume_data = self.resume_parser.parse_resume(file_obj, filename)
        ai_analysis = self.gpt4_detector.analyze_resume_for_job_roles(resume_data)
        resume_data['skills'] = ai_analysis.get('skills', [])
        return resume_data, ai_analysis
    
    def search_and_match_jobs(self, resume_data: Dict, ai_analysis: Dict, num_jobs: int = 30) -> List[Dict]:
        """Search and match jobs"""
        search_query = ai_analysis.get('primary_role', 'Professional')
        location = ai_analysis.get('location_preference', 'United States')
        
        jobs = self.job_searcher.search_jobs(keywords=search_query, location=location, limit=num_jobs)
        
        if not jobs:
            return []
        
        self.matcher.index_jobs(jobs)
        time.sleep(2)
        
        matched_jobs = self.matcher.search_similar_jobs(resume_data, ai_analysis, top_k=min(20, len(jobs)))
        matched_jobs = self._calculate_match_scores(matched_jobs, ai_analysis)
        matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return matched_jobs
    
    def _calculate_match_scores(self, jobs: List[Dict], ai_analysis: Dict) -> List[Dict]:
        """Calculate match scores"""
        candidate_skills = set([s.lower() for s in ai_analysis.get('skills', [])])
        
        for job in jobs:
            description = job.get('description', '').lower()
            title = job.get('title', '').lower()
            
            # Skill match
            matched_skills = [s for s in candidate_skills if self.matcher._fuzzy_skill_match(s, f"{title} {description}")]
            skill_match_pct = (len(matched_skills) / len(candidate_skills) * 100) if candidate_skills else 0
            
            # Semantic score
            semantic_score = job.get('similarity_score', 0)
            
            # Combined
            combined_score = 0.6 * semantic_score + 0.4 * skill_match_pct
            
            job['skill_match_percentage'] = round(skill_match_pct, 1)
            job['matched_skills'] = list(matched_skills)[:15]
            job['combined_score'] = round(combined_score, 1)
            job['overall_match'] = combined_score
            
        return jobs


# Helper functions
def extract_text_from_resume(uploaded_file):
    """Extract text from resume"""
    parser = ResumeParser()
    resume_data = parser.parse_resume(uploaded_file, uploaded_file.name)
    return resume_data['raw_text']


if __name__ == "__main__":
    print("✅ Backend module loaded successfully!")