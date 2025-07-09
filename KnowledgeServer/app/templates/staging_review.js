// staging_review.js

// --- Keyword Search Logic ---
// Artist keyword search
async function searchArtistKeyword(form, artistIdx) {
    const input = form.querySelector('input[name="add_artist_keyword"]');
    const resultsDiv = document.getElementById(`artist_kw_results_${artistIdx}`);
    const groupDiv = document.getElementById(`artist_kw_group_${artistIdx}`);
    const query = input.value.trim();
    if (!query) return;
    resultsDiv.innerHTML = '<div style="padding:8px;color:#888;">Searching...</div>';
    resultsDiv.style.display = 'block';
    try {
        const resp = await fetch('/lookup_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 8 })
        });
        const data = await resp.json();
        resultsDiv.innerHTML = '';
        if (Array.isArray(data) && data.length > 0) {
            data.forEach((kw) => {
                const div = document.createElement('div');
                div.style.padding = '6px';
                div.style.cursor = 'pointer';
                div.style.borderBottom = '1px solid #eee';
                div.innerHTML = `${kw.text} <span style='color:#888;font-size:0.9em;'>(score: ${kw.score && typeof kw.score === 'number' ? kw.score.toFixed(2) : ''})</span>`;
                div.addEventListener('click', () => addSearchedArtistKeyword(artistIdx, kw.text));
                resultsDiv.appendChild(div);
            });
        } else {
            resultsDiv.innerHTML = '<div style="padding:8px;color:#888;">No results</div>';
        }
    } catch (e) {
        resultsDiv.innerHTML = '<div style="padding:8px;color:#c00;">Error searching</div>';
    }
}

// Add selected artist keyword from search results to the dynamic checklist
function addSearchedArtistKeyword(artistIdx, kw) {
    const groupDiv = document.getElementById(`artist_kw_group_${artistIdx}`);
    // Prevent duplicates
    if ([...groupDiv.querySelectorAll('input[type=checkbox]')].some(cb => cb.value === kw)) return;
    const i = groupDiv.querySelectorAll('input[type=checkbox]').length;
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.alignItems = 'center';
    div.style.gap = '6px';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = `artist_kw_${i}`;
    checkbox.value = kw;
    checkbox.checked = true;
    const label = document.createElement('label');
    label.textContent = kw;
    div.appendChild(checkbox);
    div.appendChild(label);
    groupDiv.appendChild(div);
    document.getElementById(`artist_kw_results_${artistIdx}`).style.display = 'none';
}

function clearArtistKeywordResults(artistIdx) {
    document.getElementById(`artist_kw_results_${artistIdx}`).style.display = 'none';
}

// Artwork keyword search
async function searchArtworkKeyword(form, artistIdx, awidx) {
    const input = form.querySelector(`input[name="add_artwork_keyword_${awidx}"]`);
    const resultsDiv = document.getElementById(`artwork_kw_results_${artistIdx}_${awidx}`);
    const groupDiv = document.getElementById(`artwork_kw_group_${artistIdx}_${awidx}`);
    const query = input.value.trim();
    if (!query) return;
    resultsDiv.innerHTML = '<div style="padding:8px;color:#888;">Searching...</div>';
    resultsDiv.style.display = 'block';
    try {
        const resp = await fetch('/lookup_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 8 })
        });
        const data = await resp.json();
        resultsDiv.innerHTML = '';
        if (Array.isArray(data) && data.length > 0) {
            data.forEach((kw) => {
                const div = document.createElement('div');
                div.style.padding = '6px';
                div.style.cursor = 'pointer';
                div.style.borderBottom = '1px solid #eee';
                div.innerHTML = `${kw.text} <span style='color:#888;font-size:0.9em;'>(score: ${kw.score && typeof kw.score === 'number' ? kw.score.toFixed(2) : ''})</span>`;
                div.addEventListener('click', () => addSearchedArtworkKeyword(artistIdx, awidx, kw.text));
                resultsDiv.appendChild(div);
            });
        } else {
            resultsDiv.innerHTML = '<div style="padding:8px;color:#888;">No results</div>';
        }
    } catch (e) {
        resultsDiv.innerHTML = '<div style="padding:8px;color:#c00;">Error searching</div>';
    }
}

// Add selected artwork keyword from search results to the dynamic checklist
function addSearchedArtworkKeyword(artistIdx, awidx, kw) {
    const groupDiv = document.getElementById(`artwork_kw_group_${artistIdx}_${awidx}`);
    if ([...groupDiv.querySelectorAll('input[type=checkbox]')].some(cb => cb.value === kw)) return;
    const i = groupDiv.querySelectorAll('input[type=checkbox]').length;
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.alignItems = 'center';
    div.style.gap = '6px';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = `artwork_${awidx}_kw_${i}`;
    checkbox.value = kw;
    checkbox.checked = true;
    const label = document.createElement('label');
    label.textContent = kw;
    div.appendChild(checkbox);
    div.appendChild(label);
    groupDiv.appendChild(div);
    document.getElementById(`artwork_kw_results_${artistIdx}_${awidx}`).style.display = 'none';
}

function clearArtworkKeywordResults(artistIdx, awidx) {
    document.getElementById(`artwork_kw_results_${artistIdx}_${awidx}`).style.display = 'none';
}
