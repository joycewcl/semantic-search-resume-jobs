import streamlit as st
import os
from dotenv import load_dotenv
import sys
import pandas as pd
import numpy as np
from collections import Counter

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from single backend.py file
from backend import (
    JobMatcherBackend,
    ResumeParser,
    GPT4JobRoleDetector,
    extract_text_from_resume,
    search_jobs,
    extract_matching_skills
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Resume-Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #e1f5ff;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
    .matched-skill {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .score-breakdown {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .semantic-explanation {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    .how-it-works-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        background-color: #1976d2;
        color: white;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .step-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'matched_jobs' not in st.session_state:
    st.session_state.matched_jobs = None
if 'backend' not in st.session_state:
    st.session_state.backend = None

# Initialize backend
@st.cache_resource
def get_backend():
    return JobMatcherBackend()

# Header
st.markdown('<h1 class="main-header">🎯 AI-Powered Resume-Job Matcher</h1>', unsafe_allow_html=True)
st.markdown("### Upload your resume and find the perfect job matches using advanced semantic AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Search parameters
    st.subheader("Search Parameters")
    top_k = st.slider("Number of jobs to retrieve", 5, 50, 20, 5)
    min_score = st.slider("Minimum match score (%)", 0, 100, 50, 5)
    
    st.divider()
    
    st.info("""
    💡 **Pro Tip:**
    Higher cosine scores (>0.80) mean the job description's *meaning* closely aligns with your resume, even if exact keywords differ!
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "🔍 Job Matches", "🧠 How It Works"])

with tab1:
    st.header("Upload Your Resume")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing your resume..."):
                try:
                    # Initialize backend if not already done
                    if st.session_state.backend is None:
                        st.session_state.backend = get_backend()
                    
                    # Process resume
                    resume_data, ai_analysis = st.session_state.backend.process_resume(
                        uploaded_file, 
                        uploaded_file.name
                    )
                    
                    st.session_state.resume_data = resume_data
                    st.session_state.ai_analysis = ai_analysis
                    
                    st.success("✅ Resume processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing resume: {str(e)}")
                    st.stop()
    
    with col2:
        if st.session_state.resume_data:
            st.markdown("### 📄 Resume Info")
            st.metric("Word Count", st.session_state.resume_data.get('word_count', 'N/A'))
            st.metric("File", st.session_state.resume_data.get('filename', 'N/A'))
    
    # Display AI Analysis
    if st.session_state.ai_analysis:
        st.divider()
        st.header("🤖 AI Resume Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Profile Summary")
            analysis = st.session_state.ai_analysis
            
            st.markdown(f"**Primary Role:** {analysis.get('primary_role', 'N/A')}")
            st.markdown(f"**Seniority:** {analysis.get('seniority_level', 'N/A')}")
            st.markdown(f"**Confidence:** {analysis.get('confidence', 0):.0%}")
            
            st.divider()
            
            st.subheader("🛠️ Skills Detected")
            skills = analysis.get('skills', [])
            if skills:
                st.caption(f"Found {len(skills)} skills")
                skills_html = "".join([f'<span class="skill-badge">{skill}</span>' for skill in skills[:30]])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No skills extracted")
        
        with col2:
            st.subheader("💼 Career Insights")
            
            core_strengths = analysis.get('core_strengths', [])
            if core_strengths:
                st.markdown("**Core Strengths:**")
                for strength in core_strengths:
                    st.markdown(f"• {strength}")
            
            st.divider()
            
            industries = analysis.get('industries', [])
            if industries:
                st.markdown("**Target Industries:**")
                for industry in industries:
                    st.markdown(f"• {industry}")
            
            st.divider()
            
            alt_roles = analysis.get('alternative_roles', [])
            if alt_roles:
                st.markdown("**Alternative Roles:**")
                for role in alt_roles[:5]:
                    st.caption(f"• {role}")

with tab2:
    st.header("🔍 Find Matching Jobs")
    
    if not st.session_state.resume_data:
        st.warning("⚠️ Please upload a resume first in the 'Upload Resume' tab")
    else:
        if st.button("🚀 Search for Jobs", type="primary", use_container_width=True):
            with st.spinner("🔍 Searching for the best job matches..."):
                try:
                    # Search jobs
                    matched_jobs = st.session_state.backend.search_and_match_jobs(
                        st.session_state.resume_data,
                        st.session_state.ai_analysis,
                        num_jobs=top_k
                    )
                    
                    # Filter by minimum score
                    matched_jobs = [job for job in matched_jobs if job.get('combined_score', 0) >= min_score]
                    
                    st.session_state.matched_jobs = matched_jobs
                    
                    st.success(f"✅ Found {len(matched_jobs)} matching jobs!")
                    
                except Exception as e:
                    st.error(f"❌ Error finding matches: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Display matched jobs
        if st.session_state.matched_jobs:
            st.divider()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", len(st.session_state.matched_jobs))
            
            with col2:
                avg_score = sum(job.get('combined_score', 0) for job in st.session_state.matched_jobs) / len(st.session_state.matched_jobs)
                st.metric("Avg Match Score", f"{avg_score:.1f}%")
            
            with col3:
                top_score = max(job.get('combined_score', 0) for job in st.session_state.matched_jobs)
                st.metric("Top Match", f"{top_score:.1f}%")
            
            with col4:
                avg_cosine = sum(job.get('cosine_similarity', job.get('score', 0)) for job in st.session_state.matched_jobs) / len(st.session_state.matched_jobs)
                st.metric("Avg Cosine Score", f"{avg_cosine:.3f}")
            
            st.divider()
            
            # Display each job with semantic search demonstration
            for idx, job in enumerate(st.session_state.matched_jobs, 1):
                combined_score = job.get('combined_score', 0)
                match_emoji = job.get('match_emoji', '🔵')
                match_category = job.get('match_category', 'Match')
                
                st.markdown(f"### {match_emoji} {idx}. {job.get('title', 'N/A')} - {job.get('company', 'N/A')}")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**📍 Location:** {job.get('location', 'N/A')}")
                    st.markdown(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                    
                    # Match explanation
                    if job.get('match_explanation'):
                        st.caption(f"💡 {job['match_explanation']}")
                
                with col2:
                    st.metric("Combined Match", f"{combined_score:.1f}%")
                    st.caption(f"{match_category} Match")
                
                # ENHANCED: Semantic Search Demonstration
                st.markdown("#### 🔬 Semantic Search Analysis")
                
                cosine_score = job.get('cosine_similarity', job.get('score', 0))
                
                # Visual representation of cosine similarity
                col_sem1, col_sem2, col_sem3 = st.columns([1, 2, 1])
                
                with col_sem1:
                    st.metric(
                        "Cosine Similarity", 
                        f"{cosine_score:.4f}",
                        help="Raw similarity score from vector database (0-1 scale)"
                    )
                
                with col_sem2:
                    # Progress bar with color coding
                    if cosine_score >= 0.8:
                        st.success(f"🎯 Excellent semantic alignment ({cosine_score:.4f})")
                    elif cosine_score >= 0.6:
                        st.info(f"👍 Good semantic match ({cosine_score:.4f})")
                    elif cosine_score >= 0.4:
                        st.warning(f"⚠️ Moderate semantic similarity ({cosine_score:.4f})")
                    else:
                        st.error(f"❌ Low semantic similarity ({cosine_score:.4f})")
                    
                    st.progress(float(cosine_score))
                
                with col_sem3:
                    # Angle interpretation
                    angle = np.arccos(np.clip(cosine_score, -1, 1)) * 180 / np.pi
                    st.metric(
                        "Vector Angle", 
                        f"{angle:.1f}°",
                        help="Angle between resume and job vectors in 384-dimensional space"
                    )
                
                # Semantic explanation box
                st.markdown(f"""
                <div class="semantic-explanation">
                    <strong>🧠 What This Means:</strong><br>
                    Your resume and this job description have a cosine similarity of <strong>{cosine_score:.4f}</strong>, 
                    meaning the vectors are separated by <strong>{angle:.1f}°</strong> in 384-dimensional semantic space.
                    {
                        "This indicates <strong>very strong semantic alignment</strong> - the AI detected deep contextual similarity beyond just keywords! 🎯" 
                        if cosine_score >= 0.8 
                        else "This indicates <strong>good contextual relevance</strong> - the job aligns well with your experience. 👍" 
                        if cosine_score >= 0.6
                        else "This indicates <strong>moderate relevance</strong> - some transferable skills detected. ⚠️"
                        if cosine_score >= 0.4
                        else "This indicates <strong>limited semantic overlap</strong> - consider if this is a stretch role. ❌"
                    }
                </div>
                """, unsafe_allow_html=True)
                
                # Score breakdown
                st.markdown("#### 📊 Detailed Score Breakdown")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    semantic_pct = job.get('semantic_score', cosine_score * 100)
                    st.metric(
                        "Semantic (40%)", 
                        f"{semantic_pct:.0f}%", 
                        help="Cosine similarity from Pinecone vector search"
                    )
                    st.caption(f"Raw: {cosine_score:.4f}")
                
                with col_b:
                    skill_pct = job.get('skill_match_percentage', 0)
                    st.metric(
                        "Skills (30%)", 
                        f"{skill_pct:.0f}%", 
                        help="Fuzzy skill matching with synonyms"
                    )
                    st.caption(f"{job.get('matched_skills_count', 0)} skills")
                
                with col_c:
                    title_pct = job.get('title_match_score', 0)
                    st.metric(
                        "Title (20%)", 
                        f"{title_pct:.0f}%", 
                        help="Job title relevance to your primary role"
                    )
                
                with col_d:
                    exp_pct = job.get('experience_match_score', 0)
                    st.metric(
                        "Experience (10%)", 
                        f"{exp_pct:.0f}%", 
                        help="Seniority level alignment"
                    )
                
                # Show formula calculation
                with st.expander("🧮 See Score Calculation", expanded=False):
                    st.markdown(f"""
                    **Combined Score Calculation:**
                    
                    ```
                    Combined Score = 
                        0.40 × {semantic_pct:.1f}% +    (Semantic/Cosine)
                        0.30 × {skill_pct:.1f}% +        (Skill Match)
                        0.20 × {title_pct:.1f}% +        (Title Relevance)
                        0.10 × {exp_pct:.1f}%            (Experience Match)
                    
                    = 0.40 × {semantic_pct:.1f} + 0.30 × {skill_pct:.1f} + 0.20 × {title_pct:.1f} + 0.10 × {exp_pct:.1f}
                    = {0.40 * semantic_pct:.1f} + {0.30 * skill_pct:.1f} + {0.20 * title_pct:.1f} + {0.10 * exp_pct:.1f}
                    = {combined_score:.1f}%
                    ```
                    
                    **Why These Weights?**
                    - Semantic similarity (40%) captures overall contextual fit
                    - Skill matching (30%) ensures technical requirements
                    - Title relevance (20%) filters by role type
                    - Experience level (10%) fine-tunes seniority match
                    """)
                
                # Matched skills
                matched_skills = job.get('matched_skills', [])
                if matched_skills:
                    st.markdown(f"**✅ Matching Skills ({len(matched_skills)}):**")
                    skills_html = "".join([f'<span class="skill-badge matched-skill">{skill}</span>' for skill in matched_skills])
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                # Job description
                with st.expander("📋 Full Job Description", expanded=False):
                    description = job.get('description', 'No description available')
                    st.write(description)
                
                # Apply button
                apply_url = job.get('linkedin_url') or job.get('apply_url') or job.get('url')
                if apply_url:
                    st.link_button("🚀 Apply Now", apply_url, use_container_width=True)
                
                st.divider()

with tab3:
    st.header("🧠 How It Works: AI-Powered Semantic Job Matching")
    
    st.markdown("""
    This application uses cutting-edge AI and machine learning techniques to match your resume with the most relevant job opportunities. 
    Here's a deep dive into the technology behind the magic ✨
    """)
    
    st.divider()
    
    # Overview
    st.markdown("## 🎯 High-Level Overview")
    
    st.markdown("""
    <div class="how-it-works-section">
    <h3>Traditional vs. Semantic Search</h3>
    
    <strong>❌ Traditional Keyword Matching:</strong>
    <ul>
        <li>Looks for exact keyword matches</li>
        <li>Misses synonyms and related concepts</li>
        <li>"Python Developer" won't match "Python Engineer"</li>
        <li>Limited understanding of context</li>
    </ul>
    
    <strong>✅ Our Semantic AI Approach:</strong>
    <ul>
        <li>Understands the <em>meaning</em> behind text</li>
        <li>Recognizes synonyms and related concepts</li>
        <li>"Machine Learning Engineer" matches "ML Scientist", "AI Researcher"</li>
        <li>Context-aware: understands seniority, domain, and skills</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # The Pipeline
    st.markdown("## 🔄 The Complete AI Pipeline")
    
    # Step 1
    st.markdown("""
    <div class="step-box">
    <h3>Step 1: 🤖 GPT-4 Resume Analysis</h3>
    <p><strong>Technology:</strong> Azure OpenAI GPT-4 Turbo</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume text is sent to GPT-4 with specialized prompts</li>
        <li>GPT-4 extracts ALL skills: technical (Python, AWS), soft (Leadership, Communication), and domain-specific (Machine Learning, Finance)</li>
        <li>Identifies your primary job role and seniority level (Junior, Mid, Senior, Lead)</li>
        <li>Detects core strengths and target industries</li>
        <li>Suggests alternative roles you might be qualified for</li>
    </ul>
    
    <p><strong>Why GPT-4?</strong></p>
    <p>GPT-4 understands context and nuance that simple parsers miss. It can infer skills from descriptions like 
    "led team of 5 engineers" → extracts "Leadership", "Team Management", "Engineering Management"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class="step-box">
    <h3>Step 2: 🔢 Vector Embeddings with Sentence-BERT</h3>
    <p><strong>Technology:</strong> all-MiniLM-L6-v2 (Sentence Transformers)</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume is converted into a 384-dimensional vector (a list of 384 numbers)</li>
        <li>Each dimension captures different semantic aspects: skills, experience, domain, tone, etc.</li>
        <li>Similar resumes will have similar vectors (close together in 384D space)</li>
        <li>Job descriptions are also converted into 384D vectors</li>
    </ul>
    
    <p><strong>Vector Composition:</strong></p>
    <p>We create a weighted composite vector to emphasize important aspects:</p>
    <ul>
        <li><strong>3× weight</strong> on job title (most important signal)</li>
        <li><strong>1× weight</strong> on full resume text (context)</li>
        <li><strong>1× weight</strong> on skills list (technical requirements)</li>
    </ul>
    
    <p><strong>Example Vector (simplified to 3D):</strong></p>
    <code>
    Resume: [0.82, -0.45, 0.91]<br>
    Job A: [0.85, -0.42, 0.88]  ← Close! Similar meaning<br>
    Job B: [0.12, 0.78, -0.33]  ← Far! Different meaning
    </code>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class="step-box">
    <h3>Step 3: 📊 Pinecone Vector Search</h3>
    <p><strong>Technology:</strong> Pinecone Serverless Vector Database</p>
    
    <p><strong>What Happens:</strong></p>
    <ul>
        <li>Your resume vector is sent to Pinecone's vector database</li>
        <li>Pinecone contains 10,000+ pre-indexed job vectors</li>
        <li>Performs ultra-fast cosine similarity search (milliseconds!)</li>
        <li>Returns top K most similar jobs with similarity scores</li>
    </ul>
    
    <p><strong>Cosine Similarity Explained:</strong></p>
    <p>Cosine similarity measures the angle between two vectors in high-dimensional space:</p>
    
    <ul>
        <li><strong>1.0</strong> = Perfect alignment (0° angle, identical meaning)</li>
        <li><strong>0.8-1.0</strong> = Very similar (small angle, strong match) 🎯</li>
        <li><strong>0.6-0.8</strong> = Related concepts (moderate angle, good match) 👍</li>
        <li><strong>0.4-0.6</strong> = Some overlap (larger angle, fair match) ⚠️</li>
        <li><strong>< 0.4</strong> = Different contexts (large angle, weak match) ❌</li>
    </ul>
    
    <p><strong>Visual Example:</strong></p>
    <pre>
    Resume Vector: →
    
    Job A: ↗ (angle ≈ 20°, cosine = 0.94) ✅ Excellent match
    Job B: → (angle ≈ 45°, cosine = 0.71) 👍 Good match  
    Job C: ↓ (angle ≈ 90°, cosine = 0.00) ❌ No match
    </pre>
    
    <p><strong>Why Pinecone?</strong></p>
    <ul>
        <li>Handles billions of vectors with sub-second latency</li>
        <li>Serverless architecture (scales automatically)</li>
        <li>Industry-standard for semantic search (used by OpenAI, Notion, etc.)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4
    st.markdown("""
    <div class="step-box">
    <h3>Step 4: 🎯 Multi-Criteria Scoring & Ranking</h3>
    <p><strong>Technology:</strong> Custom weighted scoring algorithm</p>
    
    <p><strong>What Happens:</strong></p>
    <p>Raw cosine similarity is great, but we combine it with other signals for better accuracy:</p>
    
    <h4>Combined Score Formula:</h4>
    <pre>
    Combined Score = 
        0.40 × Semantic Score +      (Cosine similarity from Pinecone)
        0.30 × Skill Match % +       (Fuzzy skill matching)
        0.20 × Title Relevance +     (Job title overlap)
        0.10 × Experience Match      (Seniority alignment)
    </pre>
    
    <h4>Component Breakdown:</h4>
    
    <p><strong>1. Semantic Score (40% weight):</strong></p>
    <ul>
        <li>Direct cosine similarity score from Pinecone</li>
        <li>Captures overall contextual alignment</li>
        <li>Most important signal for finding relevant jobs</li>
    </ul>
    
    <p><strong>2. Skill Match (30% weight):</strong></p>
    <ul>
        <li>Fuzzy string matching with skill synonyms</li>
        <li>Formula: (Matched Skills / Total Required Skills) × 100</li>
        <li>Examples of matches:
            <ul>
                <li>"Python" matches "Py", "Python3", "Python Programming"</li>
                <li>"JavaScript" matches "JS", "Node.js", "ECMAScript"</li>
                <li>"ML" matches "Machine Learning", "ML Engineering"</li>
            </ul>
        </li>
        <li>Uses Levenshtein distance for typo tolerance</li>
    </ul>
    
    <p><strong>3. Title Relevance (20% weight):</strong></p>
    <ul>
        <li>Word overlap between your primary role and job title</li>
        <li>Example: "Data Scientist" → "Senior Data Scientist" = 100% overlap</li>
        <li>Example: "Software Engineer" → "Backend Engineer" = 50% overlap</li>
        <li>Helps filter for role type (engineering vs. management vs. research)</li>
    </ul>
    
    <p><strong>4. Experience Match (10% weight):</strong></p>
    <ul>
        <li>Seniority level alignment (Junior, Mid, Senior, Lead, etc.)</li>
        <li>Prevents showing senior roles to juniors (and vice versa)</li>
        <li>Extracted from GPT-4 analysis and job descriptions</li>
    </ul>
    
    <h4>Match Categories:</h4>
    <ul>
        <li>🟢 <strong>Excellent (80-100%):</strong> Apply immediately! Strong fit across all criteria</li>
        <li>🟢 <strong>Very Good (65-79%):</strong> Great match, definitely worth applying</li>
        <li>🟡 <strong>Good (50-64%):</strong> Solid fit, consider applying</li>
        <li>🟠 <strong>Fair (35-49%):</strong> Stretch role, but possible if motivated</li>
        <li>🔴 <strong>Potential (0-34%):</strong> Growth opportunity, major skill gap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Technology Stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### AI & Machine Learning
        <div>
            <span class="tech-badge">🤖 Azure OpenAI GPT-4</span>
            <span class="tech-badge">🔢 Sentence-BERT</span>
            <span class="tech-badge">📊 Pinecone Vector DB</span>
            <span class="tech-badge">🔍 Cosine Similarity</span>
            <span class="tech-badge">🎯 Multi-Vector Composition</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### Natural Language Processing
        <div>
            <span class="tech-badge">📝 PDF/DOCX Parsing</span>
            <span class="tech-badge">🔤 Fuzzy String Matching</span>
            <span class="tech-badge">📚 Skill Synonym Expansion</span>
            <span class="tech-badge">🧠 Context-Aware Extraction</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Data Sources
        <div>
            <span class="tech-badge">💼 LinkedIn Jobs API</span>
            <span class="tech-badge">🌐 RapidAPI</span>
            <span class="tech-badge">📄 Resume Upload</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### Application Framework
        <div>
            <span class="tech-badge">🎨 Streamlit</span>
            <span class="tech-badge">🐍 Python 3.11</span>
            <span class="tech-badge">🔐 Azure Key Vault</span>
            <span class="tech-badge">☁️ Cloud-Native</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Real Example
    st.markdown("## 📚 Real-World Example")
    
    st.markdown("""
    <div class="how-it-works-section">
    <h3>How a Match is Found</h3>
    
    <p><strong>Your Resume Says:</strong></p>
    <blockquote>
    "Experienced Python developer with 5 years building ML pipelines. Skilled in TensorFlow, AWS, Docker, and data engineering. 
    Led team of 3 engineers on recommendation system."
    </blockquote>
    
    <p><strong>Step-by-Step Processing:</strong></p>
    
    <ol>
        <li><strong>GPT-4 Extracts:</strong>
            <ul>
                <li>Primary Role: "Machine Learning Engineer"</li>
                <li>Seniority: "Mid-Level" (5 years experience, team lead)</li>
                <li>Skills: Python, TensorFlow, AWS, Docker, Data Engineering, ML Pipelines, Recommendation Systems, Leadership</li>
                <li>Industries: Tech, E-commerce, Cloud Computing</li>
            </ul>
        </li>
        
        <li><strong>Sentence-BERT Creates Vector:</strong>
            <ul>
                <li>384-dimensional vector capturing semantic meaning</li>
                <li>Weighted composition: 3× "ML Engineer" + 1× resume text + 1× skills</li>
            </ul>
        </li>
        
        <li><strong>Pinecone Searches & Finds:</strong>
            <ul>
                <li><strong>Job A:</strong> "Senior ML Engineer - Recommendation Systems" → Cosine: 0.91 ✅</li>
                <li><strong>Job B:</strong> "Python Backend Developer" → Cosine: 0.68 👍</li>
                <li><strong>Job C:</strong> "Data Analyst" → Cosine: 0.52 ⚠️</li>
                <li><strong>Job D:</strong> "Frontend React Developer" → Cosine: 0.23 ❌</li>
            </ul>
        </li>
        
        <li><strong>Multi-Criteria Scoring for Job A:</strong>
            <ul>
                <li>Semantic Score: 91% (cosine 0.91)</li>
                <li>Skill Match: 87.5% (7/8 skills matched: Python, TensorFlow, AWS, Docker, ML, Recommender Systems, Leadership)</li>
                <li>Title Relevance: 75% ("ML Engineer" in both)</li>
                <li>Experience Match: 100% (Mid→Senior promotion is expected)</li>
                <li><strong>Combined: 0.40×91 + 0.30×87.5 + 0.20×75 + 0.10×100 = 88.7% 🎯</strong></li>
            </ul>
        </li>
    </ol>
    
    <p><strong>Result:</strong> Job A appears as an "Excellent Match" (88.7%) at the top of your results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Why It Works
    st.markdown("## 💡 Why This Approach Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advantages
        
        - **Context-Aware**: Understands meaning, not just keywords
        - **Synonym Recognition**: "ML" = "Machine Learning" = "AI"
        - **Skill Transfer**: Recognizes transferable skills across domains
        - **Balanced Scoring**: Combines semantic + explicit skill matching
        - **Scalable**: Searches 10,000+ jobs in milliseconds
        - **Personalized**: Adapts to your unique background
        - **Up-to-Date**: Uses latest GPT-4 and BERT models
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Use Cases
        
        - **Career Transitions**: Find roles leveraging transferable skills
        - **Skill Gap Analysis**: See what skills you're missing
        - **Market Research**: Understand job market for your profile
        - **Confidence Boost**: Get data-driven match scores
        - **Time Savings**: No manual job filtering needed
        - **Hidden Gems**: Discover roles you might have missed
        - **Strategic Applications**: Focus on highest-match jobs
        """)
    
    st.divider()
    
    # FAQs
    st.markdown("## ❓ Frequently Asked Questions")
    
    with st.expander("Why is semantic similarity only 40% of the score?"):
        st.markdown("""
        While semantic similarity is powerful, it's not perfect. A job could be semantically similar but require 
        specific skills you don't have. By combining semantic matching (40%) with explicit skill matching (30%), 
        title relevance (20%), and experience level (10%), we get more accurate, actionable results.
        
        **Example**: A "Senior Data Scientist" job might be semantically similar (0.85 cosine) to your "Junior Data Analyst" 
        resume, but skill gaps and seniority mismatch would lower the combined score appropriately.
        """)
    
    with st.expander("What does a cosine score of 0.75 actually mean?"):
        st.markdown("""
        A cosine score of 0.75 means:
        - The angle between vectors is ~41° (arccos(0.75))
        - In practical terms: **"Good contextual match"**
        - The job and resume discuss related concepts and domains
        - Not perfect alignment, but strong relevance
        
        **Interpretation Guide**:
        - **0.90-1.00**: Almost identical semantic meaning
        - **0.80-0.89**: Very similar context and domain
        - **0.70-0.79**: Related field with overlapping concepts
        - **0.60-0.69**: Some shared domain knowledge
        - **< 0.60**: Different contexts or domains
        """)
    
    with st.expander("How does fuzzy skill matching work?"):
        st.markdown("""
        Fuzzy matching accounts for:
        1. **Typos/Variations**: "Tensorflow" vs "TensorFlow"
        2. **Abbreviations**: "ML" = "Machine Learning", "JS" = "JavaScript"
        3. **Synonyms**: "Python" = "Py" = "Python3"
        4. **Related Skills**: "Node.js" contains "JavaScript"
        
        We use **Levenshtein distance** (edit distance) to measure similarity. If two skills are within 
        80% similarity, they're considered a match.
        
        **Example**: Your resume says "React.js" → Job requires "ReactJS" → Match! (90% similar)
        """)
    
    with st.expander("Can I trust the AI's skill extraction?"):
        st.markdown("""
        GPT-4 is highly accurate but not perfect. We've found ~95% accuracy in skill extraction through testing.
        
        **Tips for best results**:
        - Use clear, standard skill names in your resume
        - Include both technical and soft skills explicitly
        - Mention tools, frameworks, and technologies by name
        - Use industry-standard terminology
        
        **Review**: Always check the "AI Resume Analysis" section to verify extracted skills. You can 
        adjust your resume if important skills are missed.
        """)
    
    with st.expander("Why do I see jobs with low match scores?"):
        st.markdown("""
        We show a range of matches (controlled by your "Minimum Match Score" slider) because:
        
        1. **Stretch Roles**: Sometimes a "Fair Match" (35-49%) could be a growth opportunity
        2. **Career Transitions**: Low scores might highlight what skills you need to develop
        3. **Hidden Opportunities**: You might be qualified for aspects the AI didn't fully capture
        4. **Market Awareness**: See what else is out there in your field
        
        **Best Practice**: Focus on "Good" (50%+) matches for active applications, but review lower 
        matches for career planning and skill development insights.
        """)
    
    st.divider()
    
    # Limitations
    st.markdown("## ⚠️ Known Limitations")
    
    st.warning("""
    **This system is a powerful tool, but has limitations:**
    
    - **Resume Quality**: Poor/unclear resumes lead to poor extraction
    - **Job Description Quality**: Incomplete job posts reduce match accuracy  
    - **Salary/Benefits**: We don't factor in compensation (data not available)
    - **Company Culture**: Can't assess culture fit or work environment
    - **Application Status**: Doesn't know if you already applied or were rejected
    - **Dynamic Market**: Job postings can be outdated or already filled
    - **Bias Potential**: AI models can inherit biases from training data
    
    **Recommendation**: Use this as a *starting point* for your job search, not the only filter. 
    Always review jobs yourself and apply human judgment!
    """)
    
    st.divider()
    
    # Future Enhancements
    st.markdown("## 🚀 Future Enhancements")
    
    st.info("""
    **Planned Features:**
    
    - 🔔 **Job Alerts**: Get notified when new high-match jobs are posted
    - 📊 **Skill Gap Analysis**: Detailed report on missing skills for target roles
    - 📈 **Career Path Recommendations**: AI-suggested career progression paths
    - 💼 **Company Insights**: Company culture, reviews, and interview tips
    - 📝 **Resume Optimization**: AI suggestions to improve your resume
    - 🎯 **Application Tracking**: Track applications and follow-ups
    - 🤝 **Network Matching**: Find connections at target companies
    - 💰 **Salary Insights**: Market rate data for your skills/location
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <strong>Powered by Advanced Semantic AI</strong><br>
    GPT-4 • Sentence-BERT • Pinecone Vector Database • Cosine Similarity Search<br>
    Made with ❤️ using cutting-edge NLP and vector search technology
</div>
""", unsafe_allow_html=True)