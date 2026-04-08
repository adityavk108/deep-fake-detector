# Streamlit UI Fix Plan

## Issues Identified

| # | Issue | Impact |
|---|-------|--------|
| 1 | Stale/fake instant results | Shows old cached results, not actual analysis |
| 2 | No chunk progress visibility | Can't verify multiple chunks are being processed |
| 3 | Streamlit caching interference | May reuse old results instead of fresh analysis |
| 4 | Recording audio not processed correctly | Shows "FAKE" immediately without proper analysis |
| 5 | No loading feedback | Can't confirm work is actually happening |

---

## Implementation Plan

### Phase 1: Session State & Fresh Analysis

| Step | Change | Why |
|------|--------|-----|
| 1.1 | Add `session_state.analysis_key` - a UUID that changes each time | Forces fresh run every click |
| 1.2 | Add `session_state.audio_hash` - hash of uploaded audio | Detects when audio changes |
| 1.3 | Clear previous results when new audio is uploaded | Prevents stale data display |

### Phase 2: Progress Logging & Display

| Step | Change | Why |
|------|--------|-----|
| 2.1 | Add "Processing chunk X of Y" in UI | Shows actual work happening |
| 2.2 | Add `st.progress_bar()` with chunk count | Visual progress indicator |
| 2.3 | Add logging to `detector.run()` showing each chunk | Debug trace |
| 2.4 | Print audio duration, sample rate, num chunks | Verify input is correct |

### Phase 3: Fix Recording Audio

| Step | Change | Why |
|------|--------|-----|
| 3.1 | Debug print: record audio sample rate, shape | Verify format |
| 3.2 | Add resampling if sample rate != 16000 | Model expects 16kHz |
| 3.3 | Add audio duration check before/after | Verify no truncation |
| 3.4 | Convert recording to same format as file upload | Consistent processing |

### Phase 4: Loading State Fix

| Step | Change | Why |
|------|--------|-----|
| 4.1 | Replace `st.spinner` with manual container | Full control over display |
| 4.2 | Add status message that updates during processing | "Processing chunk 3/13..." |
| 4.3 | Disable button during analysis | Prevent double-clicks |
| 4.4 | Only show results AFTER processing complete | No premature display |

### Phase 5: Debug & Verification

| Step | Change | Why |
|------|--------|-----|
| 5.1 | Add "Analysis Complete - Verified" message | Confirms real run happened |
| 5.2 | Log each step duration | Identify bottlenecks |
| 5.3 | Compare file upload vs recording processing | Ensure both work same way |

---

## Code Implementation Guide

### Session State Setup (Add to top of app_streamlit.py)

```python
# Session state initialization
if 'analysis_key' not in st.session_state:
    st.session_state.analysis_key = None
if 'last_audio_name' not in st.session_state:
    st.session_state.last_audio_name = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# When audio changes:
if audio_name != st.session_state.get('last_audio_name'):
    st.session_state.analysis_key = None
    st.session_state.analysis_results = None
```

### Analysis Button Logic

```python
if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
    # Generate fresh analysis key
    import uuid
    st.session_state.analysis_key = uuid.uuid4().hex
    st.session_state.last_audio_name = audio_name
    
    # Processing with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
        progress_bar.progress((i+1)/len(chunks))
        # ... analysis code
    
    status_text.text("✅ Analysis complete!")
```

### Disable Button During Processing

```python
if st.button("🔍 Analyze Audio", type="primary", use_container_width=True, disabled=is_processing):
    # ... code
```

---

## Files to Modify

1. `app_streamlit.py` - Main UI with session state and progress
2. `model/deepfake_detector.py` - Add chunk progress logging (optional)

---

## Success Criteria

- [ ] Results only show AFTER processing is complete
- [ ] Progress bar shows chunk X of Y during analysis
- [ ] File upload and recording produce same quality results
- [ ] No stale/cached results appearing instantly
- [ ] Button disabled during analysis prevents double-clicks
- [ ] Debug output confirms multiple chunks processed

---

## Notes

- Streamlit re-runs entire script on button click
- Session state persists across re-runs
- Use `st.empty()` for dynamic updates
- Audio files: 26s file = ~13 chunks (4s each, 50% overlap)
