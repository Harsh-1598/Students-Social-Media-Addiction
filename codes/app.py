import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils import IQRClipper
from sklearn import set_config

set_config(transform_output="pandas")

# Page config with custom theme
st.set_page_config(
    page_title="Student Addiction Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main background - Red to Black gradient */
    .stApp {
        background: linear-gradient(135deg, #8B0000 0%, #000000 35%, #000000 65%, #8B0000 100%);
    }
    
    /* Content container */
    .main .block-container {
        background-color: rgba(20, 20, 20, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(40, 40, 40, 0.3);
        transition: border-color 0.3s ease;
    }
    
    .main .block-container:hover {
        border: 1px solid rgba(139, 0, 0, 0.5);
    }
    
    /* Headers */
    h1 {
        color: #FFD700 !important;
        font-weight: 700 !important;
        text-align: center;
        padding-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    h2 {
        color: #FFA500 !important;
        font-weight: 600 !important;
        padding-top: 1rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    h3 {
        color: #FF8C00 !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    /* All text - better visibility on dark background */
    p, label, .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    /* Input labels - gold color for better visibility */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #FFD700 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Normal state */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(30, 30, 30, 0.9) !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;   /* normal border */
        border-radius: 8px !important;
        transition: border-color 0.2s ease-in-out;
    }

    /* Hover state */
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border: 1px solid #8B0000 !important;   /* dark red / gold-ish */
    }

    /* Focus (when clicked / typing) — optional but recommended */
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border: 1px solid #8B0000 !important;
        outline: none !important;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: rgba(30, 30, 30, 0.9) !important;
    }
    
    /* Metric cards - gold gradient */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #DAA520 0%, #B8860B 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(218, 165, 32, 0.4);
        border: 2px solid #FFD700;
    }
    
    div[data-testid="metric-container"] label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    
    div[data-testid="metric-container"] div {
        color: #000000 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    
    /* Buttons - Crimson/Red theme */
    .stButton > button {
        background: linear-gradient(135deg, #DC143C 0%, #8B0000 100%);
        color: #FFD700;
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 4px 15px rgba(220, 20, 60, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(220, 20, 60, 0.7);
        background: linear-gradient(135deg, #FF1744 0%, #B71C1C 100%);
        border-color: #FFD700;
    }
    
    /* Progress bar - gold */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
    }
    
    /* Dividers - consistent gold color for ALL dividers */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent 0%, #FFD700 50%, transparent 100%) !important;
        margin: 1.5rem 0 !important;
        opacity: 0.6 !important;
    }
    
    /* Override Streamlit's default divider */
    .stMarkdown hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent 0%, #FFD700 50%, transparent 100%) !important;
        margin: 1.5rem 0 !important;
        opacity: 0.6 !important;
    }
    
    /* Success boxes - dark green with gold border */
    .stSuccess {
        background-color: rgba(0, 100, 0, 0.2) !important;
        border-left: 4px solid #FFD700 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: #90EE90 !important;
    }
    
    .stSuccess p, .stSuccess li {
        color: #E0E0E0 !important;
    }
    
    /* Warning boxes - dark orange with gold border */
    .stWarning {
        background-color: rgba(139, 69, 0, 0.3) !important;
        border-left: 4px solid #FFD700 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: #FFA500 !important;
    }
    
    .stWarning p, .stWarning li {
        color: #E0E0E0 !important;
    }
    
    /* Error boxes - softer red/orange with better contrast */
    .stError {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.25) 0%, rgba(184, 134, 11, 0.15) 100%) !important;
        border-left: 5px solid #FF6347 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
        color: #FFA07A !important;
        box-shadow: 0 2px 8px rgba(255, 99, 71, 0.2) !important;
    }
    
    .stError p, .stError li {
        color: #FFE4B5 !important;
        font-weight: 500 !important;
    }
    
    .stError strong {
        color: #FFD700 !important;
    }
    
    /* Info boxes - dark blue with gold border */
    .stInfo {
        background-color: rgba(0, 0, 139, 0.3) !important;
        border-left: 4px solid #FFD700 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: #87CEEB !important;
    }
    
    .stInfo p, .stInfo li {
        color: #E0E0E0 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #A0A0A0 !important;
    }
            
    /* FIX: Remove background box from slider min/max values (they are <p> tags) */
    .stSlider p {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }

    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load(r"codes/addiction_model.pkl")

# Header
st.title("🎓 Student Social Media Addiction Analyzer")
st.markdown("<p style='text-align: center; color: #FFD700; font-size: 1.1rem; margin-top: -1rem;'>Understand your digital habits and their impact on your life</p>", unsafe_allow_html=True)
st.markdown("---")

# Section 1: Personal Information
st.subheader("👤 Personal Information")

Age = st.slider(
    "Select your age 🎂",
    min_value=10,
    max_value=30,
    value=18,
    step=1
)

# Dynamic label feedback
def age_label(age):
    if age < 13:
        return "Young Explorer 🚀"
    elif age < 18:
        return "Teen Mode ⚡"
    elif age < 23:
        return "College Zone 🎓"
    else:
        return "Adulting Phase 💼"

st.caption(f"Stage: {age_label(Age)}")


def smart_selectbox(label, options_dict):
    return st.selectbox(
        label,
        options=list(options_dict.keys()),
        format_func=lambda x: options_dict[x]
    )

Gender = smart_selectbox(
    "Gender",
    {
        "Male": "🧑 Male",
        "Female": "👩 Female"
    }
)

Academic_Level = smart_selectbox(
    "Academic Level",
    {
        "High School": "📚 High Schooler",
        "Undergraduate": "🎓 Undergrad",
        "Graduate": "🧠 Grad Scholar"
    }
)

relationship_options = {
    "Single": "Single 😎",
    "In Relationship": "Taken ❤️",
    "Complicated": "It's... complicated 😵‍💫"
}

relationship_status = st.selectbox(
    "Relationship Status",
    options=list(relationship_options.keys()),
    format_func=lambda x: relationship_options[x]
)

st.markdown("---")

# Section 2: Usage Patterns
st.subheader("📱 Digital Usage Patterns")

def smart_slider(label, min_v, max_v, default, step, help=None, tips=None):
    val = st.slider(label, min_v, max_v, default, step, help=help)
    if tips:
        for condition, text in tips:
            if condition(val):
                st.caption(text)
                break
    return val


def smart_select(label, options_dict):
    return st.selectbox(
        label,
        options=list(options_dict.keys()),
        format_func=lambda x: options_dict[x]
    )

# Avg Usage
Avg_Daily_Usage_Hours = smart_slider(
    "Average Social Media Usage 📱",
    0.0, 15.0, 3.0, 0.1,
     help="How many hours do you spend on social media per day on average? Include time on all platforms combined.",
    tips=[
        (lambda x: x < 2, "Healthy balance 👍"),
        (lambda x: x < 5, "Moderate usage 🙂"),
        (lambda x: x < 8, "Heavy scrolling 👀"),
        (lambda x: True, "Digital overload ⚠️")
    ]
)

# Platform
Most_Used_Platform = smart_select(
    "Main Platform",
    {
        "Instagram": "Instagram 📸",
        "Twitter": "Twitter 🐦",
        "TikTok": "TikTok 🎵",
        "YouTube": "YouTube ▶️",
        "Facebook": "Facebook 👥",
        "LinkedIn": "LinkedIn 💼",
        "Snapchat": "Snapchat 👻",
        "LINE": "LINE 💬",
        "KakaoTalk": "KakaoTalk 🟡",
        "VKontakte": "VKontakte 🌐",
        "WhatsApp": "WhatsApp 🟢",
        "WeChat": "WeChat 🧧"
    }
)

# Sleep
Sleep_Hours_Per_Night = smart_slider(
    "Average Sleep per Night 😴",
    2.0, 15.0, 8.0, 0.1,
    help="How many hours of sleep do you get on average each night? Quality sleep is essential for mental health, memory consolidation, and academic performance.",
    tips=[
        (lambda x: x < 5, "Severely sleep deprived 🚨"),
        (lambda x: x < 7, "Running low 😬"),
        (lambda x: x <= 9, "Healthy sleep ✅"),
        (lambda x: True, "Oversleep zone 🐻")
    ]
)

st.markdown("---")

# Section 3: Behavioral & Mental Health Factors
st.subheader("🧠 Wellbeing & Academic Impact")

Affects_Academic_Performance = st.selectbox(
    "Does social media affect your studies? 🎯",
    options=["Yes", "No"],
    help="Have you noticed that time spent on social media impacts your ability to focus on studies, complete assignments on time, attend classes, or maintain good grades?",
)

# Dynamic response feedback
if Affects_Academic_Performance == "Yes":
    st.caption("Might be worth reviewing your screen habits 📉")
else:
    st.caption("Nice — looks like you’ve got balance 👍")
Mental_Health_Score = st.slider(
    "How Are You Feeling Overall? 💭",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1,
    help="\n".join([
        "Rate your current mental vibe from 0–10:\n",
        "💀 0–2 → Crisis mode, things feel really heavy right now.\n",
        "😵 2–4 → Struggling phase, stress and mood swings showing up often.\n",
        "😐 4–6 → Mid zone, managing life but not fully thriving.\n",
        "🙂 6–8 → Stable state, handling things pretty well overall.\n",
        "😎 8–10 → Peak mode, strong mindset and feeling great."
    ])
)

Conflicts_Over_Social_Media = st.number_input(
    "Social Media–Related Conflicts (Past Month)",
    min_value=0,
    max_value=25,
    value=0,
    step=1,
    help="\n".join([
        "How many conflicts happened because of your social media use in the last month?",
        "Examples: arguments about screen time, phone use during meals, or issues caused by posts or online activity."
    ])
)
if Conflicts_Over_Social_Media == 0:
    st.caption("Peaceful month 👍")
elif Conflicts_Over_Social_Media < 5:
    st.caption("Minor friction here and there 😅")
else:
    st.caption("Might be worth reviewing habits ⚠️")

st.markdown("---")

# Prediction button
if st.button("🔍 Analyze My Digital Habits", type="primary", use_container_width=True):
    
    # Create input dataframe
    new_student = pd.DataFrame({
        "Age": [Age],
        "Gender": [Gender],
        "Academic_Level": [Academic_Level],
        "Avg_Daily_Usage_Hours": [Avg_Daily_Usage_Hours],
        "Most_Used_Platform": [Most_Used_Platform],
        "Affects_Academic_Performance": [Affects_Academic_Performance],
        "Sleep_Hours_Per_Night": [Sleep_Hours_Per_Night],
        "Mental_Health_Score": [Mental_Health_Score],
        "Relationship_Status": [relationship_status],
        "Conflicts_Over_Social_Media": [Conflicts_Over_Social_Media]
    })

    # Make prediction
    raw = model.predict(new_student)[0]
    raw = min(max(raw, 2), 9)
    percent = (raw - 2) / 7 * 100
    percent = min(max(percent, 0), 100)

    # Display results
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Risk Level", 
            value=f"{percent:.1f}%",
            help="This percentage indicates your likelihood of problematic social media use. Higher percentages suggest stronger addiction patterns."
        )
    
    with col2:
        if percent < 40:
            level = "Healthy"
            level_emoji = "🟢"
        elif percent < 70:
            level = "Moderate"
            level_emoji = "🟡"
        else:
            level = "High Risk"
            level_emoji = "🔴"
        
        st.metric(
            label="Addiction Category", 
            value=f"{level_emoji} {level}",
            help="Classification based on your usage patterns: Healthy (0-40%), Moderate (40-70%), High Risk (70-100%)"
        )
    
    with col3:
        st.metric(
            label="Severity Index", 
            value=f"{raw:.1f} / 9.0",
            help="Technical score used by the model. Higher values indicate more severe addiction patterns. Range: 2.0 (minimal) to 9.0 (severe)"
        )
    
    # Progress bar
    st.markdown("<br>", unsafe_allow_html=True)
    if percent < 40:
        bar_text = f"Risk Level: {percent:.1f}% - You're doing great! 🎉"
    elif percent < 70:
        bar_text = f"Risk Level: {percent:.1f}% - Pay attention to your habits ⚠️"
    else:
        bar_text = f"Risk Level: {percent:.1f}% - Time to make changes 🚨"
    
    st.progress(percent / 100, text=bar_text)
    
    st.markdown("---")
    
    # Detailed interpretation
    st.subheader("💡 What This Means For You")
    
    if percent < 30:
        st.success("✅ **Healthy Digital Habits** - You have excellent control over your social media use!")
        st.info("""
        **Your Current Status:**
        - You maintain strong boundaries with technology
        - Social media enhances rather than dominates your life
        - Your online habits support your goals and wellbeing
        
        **Keep It Up:**
        - Continue being mindful of your screen time
        - Maintain healthy balance between online and offline life
        - Share your healthy habits with friends who might struggle
        """)
    
    elif percent < 50:
        st.info("ℹ️ **Watch Zone** - Your usage is manageable but starting to show early warning signs.")
        st.warning("""
        **What's Happening:**
        - You sometimes spend more time online than planned
        - Occasional difficulty putting the phone down
        - Social media might distract you from important tasks
        
        **Action Steps:**
        - Set daily time limits (2-3 hours maximum)
        - Use screen time tracking on your phone
        - Create "phone-free zones" during study time
        - Replace scrolling with physical activities
        - Turn off non-urgent notifications
        """)
    
    elif percent < 70:
        st.warning("⚠️ **Concerning Pattern** - Social media is likely interfering with important areas of your life.")
        st.error("""
        **Warning Signs:**
        - Affecting academic performance
        - Anxiety when unable to access phone
        - Relationships or responsibilities suffering
        - Sleep quality disrupted
        
        **Take Action:**
        - Strict limit: 1-2 hours daily maximum
        - Delete apps; use web versions only
        - Use app blockers during study hours
        - Phone stays outside bedroom at night
        - Replace scrolling with exercise, reading, or hobbies
        - Talk to a counselor about digital wellness
        """)
    
    else:
        st.error("🚨 **Critical Level** - Immediate intervention required.")
        st.error("""
        **Critical Indicators:**
        - Unable to control usage despite negative consequences
        - Academic performance has declined significantly
        - Important relationships are strained or damaged
        - Mental wellbeing is deteriorating
        - Physical health showing signs of impact
        
        **URGENT Actions Required:**
        
        **Get Professional Help:**
        - 📞 Contact your college/university counseling center immediately
        - 🏥 Consider therapy specializing in behavioral addiction
        - 👥 Join support groups for digital addiction
        
        **Immediate Reset Plan:**
        - Remove all social media apps from your phone today
        - Hand over account passwords to a trusted friend/family member
        - Limit usage to 30 minutes daily (web browser only, not apps)
        - Keep phone outside bedroom - use a traditional alarm clock
        - Replace social media time with: exercise, face-to-face socializing, hobbies, or learning
        
        **Address Root Causes:**
        - Work with a therapist to understand underlying issues (loneliness, anxiety, depression, low self-esteem)
        - Build healthy coping mechanisms for stress and emotions
        - Develop real-world connections and find purpose beyond social media
        
        **Remember:** Recovery is absolutely possible. Thousands have successfully overcome this. You're not alone - reach out for support today.
        """)
    
    st.markdown("---")
    
    # Contributing factors analysis
    st.subheader("🔍 Detailed Habit Analysis")
    
    factors = []
    positive_factors = []
    
    # Analyze negative factors
    if Avg_Daily_Usage_Hours > 6:
        severity = "very high" if Avg_Daily_Usage_Hours > 10 else "high"
        factors.append(f"🔴 **Excessive screen time**: {Avg_Daily_Usage_Hours:.1f} hours/day is {severity}. Recommended: under 2 hours.")
    elif Avg_Daily_Usage_Hours > 3:
        factors.append(f"🟡 **Above average**: {Avg_Daily_Usage_Hours:.1f} hours/day. Try reducing to 2-3 hours max.")
    
    if Sleep_Hours_Per_Night < 6:
        factors.append(f"🔴 **Sleep deprivation**: {Sleep_Hours_Per_Night:.1f} hours is critically low. Aim for 7-9 hours.")
    elif Sleep_Hours_Per_Night < 7:
        factors.append(f"🟡 **Insufficient sleep**: {Sleep_Hours_Per_Night:.1f} hours. Increase to 7-9 hours.")
    
    if Mental_Health_Score < 4:
        factors.append(f"🔴 **Wellbeing concern**: {Mental_Health_Score:.1f}/10. Please seek mental health support.")
    elif Mental_Health_Score < 6:
        factors.append(f"🟡 **Moderate wellbeing**: {Mental_Health_Score:.1f}/10. Focus on self-care.")
    
    if Conflicts_Over_Social_Media > 10:
        factors.append(f"🔴 **Frequent conflicts**: {Conflicts_Over_Social_Media} arguments/month. Social media is damaging relationships.")
    elif Conflicts_Over_Social_Media > 5:
        factors.append(f"🟡 **Relationship strain**: {Conflicts_Over_Social_Media} conflicts/month.")
    elif Conflicts_Over_Social_Media > 2:
        factors.append(f"🟡 **Some tension**: {Conflicts_Over_Social_Media} conflicts creating friction.")
    
    if Affects_Academic_Performance == "Yes":
        factors.append("🔴 **Academic impact**: Social media is hurting your grades.")
    
    # Analyze positive factors
    if Avg_Daily_Usage_Hours <= 2:
        positive_factors.append(f"✅ **Excellent screen time**: {Avg_Daily_Usage_Hours:.1f} hours/day is healthy!")
    elif Avg_Daily_Usage_Hours <= 3:
        positive_factors.append(f"✅ **Good usage**: {Avg_Daily_Usage_Hours:.1f} hours/day is within limits.")
    
    if Sleep_Hours_Per_Night >= 8:
        positive_factors.append(f"✅ **Optimal sleep**: {Sleep_Hours_Per_Night:.1f} hours supports health!")
    elif Sleep_Hours_Per_Night >= 7:
        positive_factors.append(f"✅ **Adequate sleep**: {Sleep_Hours_Per_Night:.1f} hours is good.")
    
    if Mental_Health_Score >= 8:
        positive_factors.append(f"✅ **Strong wellbeing**: {Mental_Health_Score:.1f}/10 - excellent!")
    elif Mental_Health_Score >= 7:
        positive_factors.append(f"✅ **Good wellbeing**: {Mental_Health_Score:.1f}/10 - solid.")
    
    if Conflicts_Over_Social_Media == 0:
        positive_factors.append("✅ **Healthy relationships**: No conflicts!")
    elif Conflicts_Over_Social_Media <= 1:
        positive_factors.append("✅ **Minimal conflict**: Very few arguments.")
    
    if Affects_Academic_Performance == "No":
        positive_factors.append("✅ **Academics protected**: Studies not suffering.")
    
    # Display factors
    if factors:
        st.warning("**⚠️ Areas That Need Attention:**")
        for factor in factors:
            st.markdown(f"- {factor}")
        st.markdown("")
    
    if positive_factors:
        st.success("**✅ What You're Doing Well:**")
        for factor in positive_factors:
            st.markdown(f"- {factor}")
        st.markdown("")
    
    if not factors:
        st.success("🎉 **Excellent!** All your digital habits are healthy!")
    
    # Resources section
    st.markdown("---")
    st.subheader("📚 Helpful Resources & Support")
    
    st.info("""
    **🇮🇳 India-Based Mental Health & Addiction Support:**
    
    **24/7 Helplines:**
    - 📞 **Vandrevala Foundation**: 1860-2662-345 / 1800-2333-330 (24/7, Free)
    - 📞 **iCall (TISS)**: 9152987821 (Mon-Sat, 8 AM - 10 PM)
    - 📞 **NIMHANS Helpline**: 080-46110007 (Mon-Sat, 9 AM - 5:30 PM)
    - 📞 **Sneha India**: 044-24640050 (24/7, Chennai-based)
    - 📞 **Fortis Stress Helpline**: 8376804102 (24/7)
    - 📱 **WhatsApp Support - YourDOST**: +91-8448448442
    
    **Student-Focused Services:**
    - 🎓 **Your College/University Counseling Center** (free for students)
    - 💻 **YourDOST**: www.yourdost.com (online counseling for students)
    - 💻 **InnerHour**: www.theinnerhour.com (affordable therapy)
    - 💻 **Wysa**: AI chatbot + therapist support
    
    **Apps for Digital Wellness:**
    - 📱 **Forest** - Gamified focus timer (helps reduce phone usage)
    - 📱 **StayFree** - Screen time tracker with limits (Indian app)
    - 📱 **ActionDash** - Digital wellbeing & screen time control
    - 📱 **Headspace** / **Calm** - Meditation for stress management
    
    **Educational Resources:**
    - 📚 "Digital Minimalism" by Cal Newport
    - 📚 "How to Break Up with Your Phone" by Catherine Price
    - 📚 "Irresistible" by Adam Alter
    - 🌐 **White Swan Foundation**: www.whiteswanfoundation.org (mental health resources)
    
    **Government Resources:**
    - 🏛️ **Tele-MANAS**: 14416 (National Mental Health Helpline)
    - 🏛️ **Manashakti**: Karnataka Govt Mental Health Initiative
    
    **Remember:** Seeking help is a sign of strength, not weakness. All services listed are confidential and professional.
    """)

st.markdown("---")
st.subheader("ℹ️ Prediction Accuracy Notice")

st.markdown("""
This model performs best when inputs fall within the ranges observed in the training data.
Predictions outside these ranges are still provided but should be interpreted with caution.
""")

st.markdown("""
**Most reliable input ranges:**
- 🎂 **Age:** 18 – 24 years  
- 📱 **Daily social media usage:** 1.5 – 8.5 hours  
- 😴 **Sleep duration:** 3.8 – 9.6 hours/night  
- 🧠 **Mental health score:** 4 – 9 (out of 10)  
- ⚡ **Monthly social media conflicts:** 0 – 5
""")

# Footer
st.markdown("---")
st.caption("📌 **Important:** This tool provides educational insights based on research patterns. It is NOT a clinical diagnosis and should not replace professional medical or psychological evaluation.")
st.caption("💡 **How It Works:** Our AI model analyzes your responses against patterns identified in thousands of student cases to estimate addiction risk levels.")
st.caption("🔒 **Your Privacy:** All analysis happens instantly in your browser. No data is collected, stored, or shared with anyone. Your information remains completely private.")
