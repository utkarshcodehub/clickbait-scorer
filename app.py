import streamlit as st
import joblib
import pandas as pd
import scipy.sparse as sp
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import io

# Load
model = joblib.load('clickbait_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def extract_features(headline):
    h = headline.lower()
    return {
        'has_number': bool(re.search(r'\d+', headline)),
        'all_caps_words': sum(1 for w in headline.split() if w.isupper() and len(w) > 2),
        'has_question': '?' in headline,
        'has_exclamation': '!' in headline,
        'word_count': len(headline.split()),
        'has_you': 'you' in h or 'your' in h,
        'has_shocking': any(w in h for w in ['shock', 'stun', 'amaz', 'unbeliev', 'reveal', 'secret', 'truth', 'real reason', 'dark', 'exposes']),
        'has_superlative': any(w in h for w in ['best', 'worst', 'most', 'least', 'never', 'always', 'every', 'only']),
        'has_will': 'will' in h,
        'has_how_why': headline.lower().startswith(('how', 'why', 'what')),
        'headline_len': len(headline),
        'has_ellipsis': '...' in headline,
    }

def extract_headline_from_url(url):
    """Scrape headline from a news article URL with cloud-safe fallbacks"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    try:
        r = requests.get(url.strip(), headers=headers, timeout=10, allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Try in order of reliability
        # 1. og:title (most consistent across sites)
        og = soup.find('meta', property='og:title')
        if og and og.get('content', '').strip():
            return og['content'].strip()
        
        # 2. twitter:title
        tw = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw and tw.get('content', '').strip():
            return tw['content'].strip()
        
        # 3. h1
        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        
        # 4. title tag, strip site name
        title = soup.find('title')
        if title:
            t = title.get_text(strip=True)
            # Remove site name suffix (e.g. "Headline | NDTV" → "Headline")
            for sep in [' | ', ' - ', ' – ', ' — ']:
                if sep in t:
                    t = t.split(sep)[0].strip()
            if len(t) > 20:
                return t
        
        return None
    
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.SSLError:
        # Retry without SSL verification (some Indian news sites have cert issues)
        try:
            r = requests.get(url.strip(), headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(r.text, 'html.parser')
            og = soup.find('meta', property='og:title')
            if og and og.get('content', '').strip():
                return og['content'].strip()
            h1 = soup.find('h1')
            if h1:
                return h1.get_text(strip=True)
            return None
        except:
            return None
    except Exception:
        return None


def get_clickbait_dna(headline):
    """Returns list of (word/phrase, reason, severity) tuples found in headline"""
    h_lower = headline.lower()
    found = []
    
    PATTERNS = [
        # (pattern_list, label, severity)  severity: 'high' | 'medium' | 'low'
        (['won\'t believe', 'you won\'t', 'will shock you', 'left speechless', 'can\'t handle'], 
         'Curiosity gap phrase', 'high'),
        
        (['shocking', 'shocked', 'shocks', 'stuns', 'stunned', 'stunning'], 
         'Shock language', 'high'),
        
        (['secret', 'secrets', 'hidden', 'revealed', 'exposes', 'exposed', 'real truth', 'dark truth', 'real reason'], 
         'False exclusivity', 'high'),
        
        (['you won\'t', 'you need to', 'you should', 'your life', 'you are', 'you have'], 
         'Direct address manipulation', 'medium'),
        
        (['never', 'always', 'every', 'only one', 'no one', 'everyone'], 
         'Absolute language', 'medium'),
        
        (['best', 'worst', 'most', 'least', 'biggest', 'smallest', 'greatest'], 
         'Superlative bait', 'medium'),
        
        (['before it\'s too late', 'before you', 'this changes everything', 'game changer', 'life changing'], 
         'Urgency trigger', 'high'),
        
        (['here\'s why', 'here\'s how', 'here\'s what', 'this is why', 'this is how'], 
         'Curiosity setup', 'medium'),
        
        (['watch:', 'video:', 'photos:', 'pics:', 'see:'], 
         'Multimedia bait', 'low'),
        
        (['rs ', '₹', 'crore', 'lakh'], 
         'Money bait', 'low'),
    ]
    
    for pattern_list, label, severity in PATTERNS:
        for pattern in pattern_list:
            if pattern in h_lower:
                # Find actual position in original headline (case-insensitive)
                idx = h_lower.find(pattern)
                actual = headline[idx:idx+len(pattern)]
                found.append({'text': actual, 'label': label, 'severity': severity, 'start': idx, 'end': idx+len(pattern)})
                break  # one match per category is enough
    
    # Check for ALL CAPS words separately
    import re
    caps_words = re.findall(r'\b[A-Z]{2,}\b', headline)
    for word in caps_words:
        if word not in ['BJP', 'RSS', 'RBI', 'UPI', 'IPL', 'IIT', 'CBI', 'ED', 'PM', 'CM', 'UP', 'MP', 'UK', 'US', 'UN', 'GDP', 'FIR', 'ISRO', 'NASA', 'WHO', 'IMF', 'CAA', 'NRC', 'SC', 'HC']:
            idx = headline.find(word)
            found.append({'text': word, 'label': 'Emphasis caps', 'severity': 'high', 'start': idx, 'end': idx+len(word)})
    
    # Check for numbers in "X things/reasons/signs" pattern
    list_match = re.search(r'\b(\d+)\s+(things|reasons|signs|ways|facts|secrets|tricks)\b', h_lower)
    if list_match:
        idx = h_lower.find(list_match.group())
        actual = headline[idx:idx+len(list_match.group())]
        found.append({'text': actual, 'label': 'Listicle hook', 'severity': 'medium', 'start': idx, 'end': idx+len(list_match.group())})
    
    return found

def render_dna_highlight(headline, dna_matches):
    """Render headline with colored highlights using HTML"""
    if not dna_matches:
        return f'<span style="font-size:18px">{headline}</span>'
    
    COLORS = {
        'high':   ('#ff4444', '#fff0f0'),   # red text, light red bg
        'medium': ('#ff8800', '#fff8f0'),   # orange text, light orange bg
        'low':    ('#888800', '#fffff0'),   # olive text, light yellow bg
    }
    
    # Sort matches by start position, remove overlaps
    sorted_matches = sorted(dna_matches, key=lambda x: x['start'])
    non_overlapping = []
    last_end = 0
    for m in sorted_matches:
        if m['start'] >= last_end:
            non_overlapping.append(m)
            last_end = m['end']
    
    result = ''
    cursor = 0
    for m in non_overlapping:
        # Add unhighlighted text before this match
        result += headline[cursor:m['start']]
        # Add highlighted match
        text_color, bg_color = COLORS[m['severity']]
        label = m['label']
        text = m['text']
        result += (
            f'<span style="background:{bg_color}; color:{text_color}; '
            f'font-weight:600; border-radius:3px; padding:1px 4px; '
            f'border-bottom:2px solid {text_color}; cursor:help;" '
            f'title="{label}">{text}</span>'
        )
        cursor = m['end']
    
    result += headline[cursor:]
    return f'<span style="font-size:18px; line-height:1.6">{result}</span>'


def generate_share_card(headline, score, verdict, dna_matches):
    """Generate a shareable PNG card for the score result"""
    
    # Canvas
    W, H = 800, 420
    
    # Colors based on verdict
    if score > 0.65:
        accent = (220, 50, 50)       # red
        bg_top = (40, 10, 10)
        label = "CLICKBAIT"
    elif score > 0.35:
        accent = (220, 140, 30)      # orange
        bg_top = (40, 30, 10)
        label = "BORDERLINE"
    else:
        accent = (40, 180, 100)      # green
        bg_top = (10, 35, 20)
        label = "FACTUAL"
    
    bg_bottom = (18, 18, 22)
    
    img = Image.new('RGB', (W, H), bg_bottom)
    draw = ImageDraw.Draw(img)
    
    # Top accent bar
    draw.rectangle([0, 0, W, 6], fill=accent)
    
    # Gradient-like top section background
    draw.rectangle([0, 6, W, 160], fill=bg_top)
    
    # Score circle (manual)
    cx, cy, r = 100, 90, 55
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=bg_bottom, outline=accent, width=4)
    
    # Fonts — use default since we can't guarantee custom fonts on all systems
    try:
        font_big   = ImageFont.truetype("arialbd.ttf", 36)
        font_med   = ImageFont.truetype("arialbd.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf",   14)
        font_tiny  = ImageFont.truetype("arial.ttf",   12)
    except:
        font_big   = ImageFont.load_default()
        font_med   = font_big
        font_small = font_big
        font_tiny  = font_big
    
    # Score text inside circle
    score_text = f"{score:.0%}"
    bbox = draw.textbbox((0,0), score_text, font=font_big)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw//2, cy - th//2 - 2), score_text, font=font_big, fill=accent)
    
    # Verdict label
    draw.text((175, 55), label, font=font_big, fill=accent)
    draw.text((175, 100), verdict.replace('🚨','').replace('⚠️','').replace('✅','').strip(), 
              font=font_small, fill=(160, 160, 170))
    
    # Divider
    draw.line([30, 165, W-30, 165], fill=(50, 50, 60), width=1)
    
    # Headline text (wrapped)
    margin = 40
    max_chars_per_line = 72
    words = headline.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars_per_line:
            current = (current + " " + word).strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    
    y = 185
    for line in lines[:3]:  # max 3 lines
        draw.text((margin, y), line, font=font_med, fill=(230, 230, 235))
        y += 30
    if len(lines) > 3:
        draw.text((margin, y), "...", font=font_med, fill=(130, 130, 140))
        y += 30
    
    # DNA signals
    y += 10
    draw.line([30, y, W-30, y], fill=(40, 40, 50), width=1)
    y += 14
    
    draw.text((margin, y), "SIGNALS DETECTED", font=font_tiny, fill=(100, 100, 110))
    y += 20
    
    signal_colors = {'high': (220, 80, 80), 'medium': (220, 150, 50), 'low': (180, 180, 80)}
    
    if dna_matches:
        # Show max 3 signals in two columns
        col2_x = W // 2 + 10
        for i, match in enumerate(dna_matches[:4]):
            col_x = margin if i % 2 == 0 else col2_x
            row_y = y + (i // 2) * 22
            color = signal_colors.get(match['severity'], (150, 150, 150))
            dot = "● "
            draw.text((col_x, row_y), dot, font=font_tiny, fill=color)
            signal_text = f"{match['text'][:22]}  –  {match['label']}"
            draw.text((col_x + 14, row_y), signal_text, font=font_tiny, fill=(160, 160, 170))
    else:
        draw.text((margin, y), "No clickbait signals detected", font=font_tiny, fill=(100, 180, 120))
    
    # Footer
    draw.rectangle([0, H-38, W, H], fill=(12, 12, 15))
    draw.text((margin, H-25), "Indian News Clickbait Scorer", font=font_tiny, fill=(80, 80, 90))
    draw.text((W-180, H-25), "Built with Python + Streamlit", font=font_tiny, fill=(60, 60, 70))
    
    # Convert to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def score_headline(headline):
    feats = pd.DataFrame([extract_features(headline)])
    tfidf_vec = vectorizer.transform([headline])
    x = sp.hstack([tfidf_vec, sp.csr_matrix(feats.values.astype(float))])
    prob = model.predict_proba(x)[0][1]
    return prob

def get_reasons(headline, features):
    reasons = []
    if features['all_caps_words'] > 0:
        reasons.append("🔴 Contains ALL CAPS words (classic clickbait signal)")
    if features['has_shocking']:
        reasons.append("🔴 Uses emotionally charged language (shock, reveal, truth, secret...)")
    if features['has_you']:
        reasons.append("🟡 Speaks directly to 'you' — personalisation trick")
    if features['has_number']:
        reasons.append("🟡 Contains a number — listicle formatting")
    if features['has_question']:
        reasons.append("🟡 Ends/uses a question — creates curiosity gap")
    if features['has_superlative']:
        reasons.append("🟡 Uses superlatives (best/worst/never/always)")
    if features['has_ellipsis']:
        reasons.append("🔴 Uses '...' — unfinished thought hook")
    if not reasons:
        reasons.append("✅ No strong clickbait signals detected")
    return reasons

st.set_page_config(page_title="Clickbait Scorer", page_icon="📰", layout="centered")

#Hero section
st.markdown("""
            <div style="padding: 1.5rem 0 0.5rem">
    <h1 style="margin-bottom: 0.3rem">📰 Indian News Clickbait Scorer</h1>
    <p style="font-size: 1.1rem; color: #888; margin-bottom: 0.8rem">
        Indian news headlines are engineered to manipulate. This tool scores any headline 
        and shows you <em>exactly</em> which words are doing the manipulating — and why.
    </p>
    <p style="font-size: 0.95rem; color: #666">
        Paste a headline → get a clickbait score → see the DNA breakdown of what triggered it.
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()
st.title("📰 Indian News Clickbait Scorer")
st.caption("Paste any Indian news headline. Get a clickbait score + breakdown of why.")

# Example button
EXAMPLES = [
    "You Won't BELIEVE What This IIT Student Did That Shocked His Professors",
    "10 Shocking Secrets About Indian Politics No One Wants You To Know",
    "Watch: Moment That Left The Entire Nation Speechless",
    "BJP Leader EXPOSES Dark Truth About Opposition Before It's Too Late",
]

st.markdown("**Try an example:**")
ex_cols = st.columns(len(EXAMPLES))
example_chosen = ""
for i, ex in enumerate(EXAMPLES):
    with ex_cols[i]:
        short_label = ex[:28] + "..."
        if st.button(short_label, key=f"ex_{i}", use_container_width=True):
            example_chosen = ex

st.markdown("")

headline = st.text_area(
    "Paste headline here:", 
    value= example_chosen, 
    height=80, 
    placeholder="e.g. You Won't Believe What This IIT Student Did..."
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("Analyze", use_container_width=True)

if analyze and headline.strip():
    score = score_headline(headline)
    features = extract_features(headline)
    
    st.divider()
    
    # Score display
    if score < 0.35:
        verdict = "✅ Factual"
        color = "green"
        desc = "This reads like a straightforward news headline."
    elif score < 0.65:
        verdict = "⚠️ Borderline"
        color = "orange"
        desc = "Some clickbait signals, but not extreme."
    else:
        verdict = "🚨 Clickbait"
        color = "red"
        desc = "High clickbait probability. Strong manipulative signals detected."
    
    st.markdown(f"### {verdict}")
    st.progress(score)
    st.markdown(f"**Score: {score:.0%}** clickbait probability — {desc}")
    
    #DNA Breakdown
    st.markdown("#### 🧬 Clickbait DNA")
    dna = get_clickbait_dna(headline)

    # Highlighted headline
    st.markdown("**Highlighted headline:**")
    highlighted = render_dna_highlight(headline, dna)
    st.markdown(highlighted, unsafe_allow_html=True)
    st.caption("Hover over highlighted words to see what signal they trigger.")

    # Legend
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.markdown('<span style="color:#ff4444">🔴 High signal</span>', unsafe_allow_html=True)
    col2.markdown('<span style="color:#ff8800">🟠 Medium signal</span>', unsafe_allow_html=True)
    col3.markdown('<span style="color:#888800">🟡 Low signal</span>', unsafe_allow_html=True)

    # Signal list
    if dna:
        st.markdown("**Signals detected:**")
        for match in dna:
            severity_icon = {'high': '🔴', 'medium': '🟠', 'low': '🟡'}[match['severity']]
            st.markdown(f"- {severity_icon} **\"{match['text']}\"** — {match['label']}")
    else:
        st.markdown("✅ No clickbait signals found in the text.")

    st.divider()

    # Share card
    st.markdown("#### 🖼️ Share Card")
    card_buf = generate_share_card(headline, score, verdict, dna)
    st.image(card_buf, use_column_width=True)

    # Reset buffer for download
    card_buf.seek(0)
    st.download_button(
        label="⬇️ Download card",
        data=card_buf,
        file_name="clickbait_score.png",
        mime="image/png"
    )

    # Old reasons (keep as secondary)
    with st.expander("See full feature breakdown"):
        for reason in get_reasons(headline, features):
            st.markdown(f"- {reason}")


# Batch mode
with st.expander("📋 Batch mode — score multiple headlines at once"):
    batch = st.text_area("One headline per line:", height=150)
    if st.button("Score All"):
        lines = [l.strip() for l in batch.split('\n') if l.strip()]
        if lines:
            results = [{'Headline': h, 'Score': f"{score_headline(h):.0%}", 
                       'Verdict': '🚨 Clickbait' if score_headline(h) > 0.65 else ('⚠️ Borderline' if score_headline(h) > 0.35 else '✅ Factual')} 
                      for h in lines]
            st.dataframe(pd.DataFrame(results), use_container_width=True)

#Bulk URL
st.divider()
with st.expander("📋 Bulk URL Scorer - score multiple articles at once"):
  st.caption("Paste one news article URL per line. It'll extract and score each headline automatically.")

  bulk_urls = st.text_area("Paste URLs here (one per line):", height=150, 
                            placeholder="https://timesofindia.indiatimes.com/...\nhttps://ndtv.com/...\nhttps://thewire.in/...")

  if st.button("Score All URLs"):
      urls = [u.strip() for u in bulk_urls.strip().split('\n') if u.strip()]
      if not urls:
          st.warning("Paste at least one URL.")
      else:
          results = []
          progress = st.progress(0)
          status = st.empty()
          
          for i, url in enumerate(urls):
              status.text(f"Scraping {i+1}/{len(urls)}...")
              headline = extract_headline_from_url(url)
              
              if headline:
                  s = score_headline(headline)
                  if s > 0.65:
                      verdict = "🚨 Clickbait"
                  elif s > 0.35:
                      verdict = "⚠️ Borderline"
                  else:
                      verdict = "✅ Factual"
                  
                  # Detect source from URL
                  source = url.split('/')[2].replace('www.', '').split('.')[0].upper()
                  
                  results.append({
                      'Source': source,
                      'Headline': headline,
                      'Score': f"{s:.0%}",
                      'Verdict': verdict,
                      '_score_raw': s,
                      'URL': url
                  })
              else:
                  results.append({
                      'Source': url.split('/')[2].replace('www.', '').split('.')[0].upper(),
                      'Headline': '⚠️ Could not extract headline',
                      'Score': '—',
                      'Verdict': '—',
                      '_score_raw': 0,
                      'URL': url
                  })
              
              progress.progress((i + 1) / len(urls))
          
          status.empty()
          progress.empty()
          
          if results:
              df_results = pd.DataFrame(results)
              
              # Sort by score descending
              df_display = df_results.sort_values('_score_raw', ascending=False)
              
              # Show table
              st.dataframe(
                  df_display[['Source', 'Headline', 'Score', 'Verdict']],
                  use_container_width=True,
                  hide_index=True
              )
              
              # Summary stats
              valid = [r for r in results if r['Score'] != '—']
              if valid:
                  avg = sum(r['_score_raw'] for r in valid) / len(valid)
                  most_clickbaity = max(valid, key=lambda x: x['_score_raw'])
                  least_clickbaity = min(valid, key=lambda x: x['_score_raw'])
                  
                  col1, col2, col3 = st.columns(3)
                  col1.metric("Average clickbait score", f"{avg:.0%}")
                  col2.metric("Most clickbaity", most_clickbaity['Source'], most_clickbaity['Score'])
                  col3.metric("Least clickbaity", least_clickbaity['Source'], least_clickbaity['Score'])
