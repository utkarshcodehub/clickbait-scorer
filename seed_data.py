import pandas as pd

CLICKBAIT = [
    "You Won't Believe What This IIT Student Did That Shocked His Professors",
    "THIS Is Why Indians Are So Different From The Rest Of The World",
    "10 Signs Your Partner Is Secretly Cheating On You (Number 7 Will Shock You)",
    "What Happened Next Will Leave You Speechless",
    "Bollywood Actress EXPOSES Dark Secret That Changes Everything",
    "The Real Reason Why India Will NEVER Become A Superpower",
    "This One Trick Will Make You Rich In 30 Days, Bankers Don't Want You To Know",
    "Indians Are Doing THIS In America And Americans Can't Handle It",
    "Shocking Truth About Your Favourite Celebrity REVEALED",
    "Why Every Indian Should Know THIS Secret Before It's Too Late",
    "The REAL Reason Modi Is Doing THIS Will Blow Your Mind",
    "Watch: Moment That Made The Entire Nation Cry",
    "This 22-Year-Old Made ₹1 Crore In 3 Months And Here's Exactly How",
    "Scientists Are Baffled By What This Indian Village Has Been Hiding For Centuries",
    "The Dark Truth About Indian Education That No One Wants To Admit",
]

FACTUAL = [
    "Supreme Court Issues Notice to Centre on Electoral Bond Petitions",
    "India's GDP Growth Projected at 6.8% for FY2025, IMF Says",
    "Parliament Passes New Data Protection Bill After Three-Year Deliberation",
    "Cyclone Remal Makes Landfall in West Bengal, Red Alert in Three Districts",
    "ISRO Successfully Tests Cryogenic Engine for Gaganyaan Mission",
    "RBI Keeps Repo Rate Unchanged at 6.5% for Sixth Consecutive Meeting",
    "India Signs Free Trade Agreement with European Union After 16 Years of Talks",
    "Delhi Air Quality Index Crosses 400 for Third Consecutive Day",
    "New Study Finds Microplastics in Human Blood Samples Across 22 Indian Cities",
    "Manipur Violence: 14 Dead in Fresh Clashes, Army Deployed in Four Districts",
    "India Abstains on UN Vote on Gaza Ceasefire Resolution",
    "Union Budget 2025: Key Allocations for Defence, Education and Infrastructure",
    "Wheat Production Expected to Drop 4% Due to Unseasonal Rains in Punjab",
    "CBI Files Chargesheet in ₹3,800 Crore Scam Linked to Former State Minister",
    "IIT Bombay Researchers Develop Low-Cost Water Purification Using Rice Husk",
]

df_seed = pd.DataFrame(
    [{'headline': h, 'label': 1} for h in CLICKBAIT] +
    [{'headline': h, 'label': 0} for h in FACTUAL]
)
df_seed.to_csv('seed_data.csv', index=False)
print(f"Seed: {len(df_seed)} rows")