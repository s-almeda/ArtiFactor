function escapeHtml(text) {
    var map = {
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

function renderTable(fields, rows, tableId) {
    if (!rows.length) return '<em>No rows</em>';
    let html = '<table class="sql-table" id="' + tableId + '"><thead><tr>';
    for (const f of fields) html += '<th>' + escapeHtml(f) + '</th>';
    html += '</tr></thead><tbody>';
    for (const row of rows) {
        html += '<tr>';
        for (const cell of row) {
            let cellVal = cell;
            // Try to pretty-print JSON
            if (typeof cellVal === 'string') {
                try {
                    const parsed = JSON.parse(cellVal);
                    if (typeof parsed === 'object') {
                        cellVal = '<pre>' + escapeHtml(JSON.stringify(parsed, null, 2)) + '</pre>';
                    } else {
                        cellVal = escapeHtml(cellVal);
                    }
                } catch (e) {
                    cellVal = escapeHtml(cellVal);
                }
            }
            html += '<td>' + cellVal + '</td>';
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
}

window.renderSqlTabs = function(data) {
    const container = document.getElementById('sql-commands-container');
    let html = `<div class="sql-tabs">
        <button class="sql-tab-btn" onclick="showSqlTab('raw')">Raw SQL</button>
        <button class="sql-tab-btn" onclick="showSqlTab('text')">Text Entries Table</button>
        <button class="sql-tab-btn" onclick="showSqlTab('image')">Image Entries Table</button>
    </div>`;
    html += `<div id="sql-tab-raw" class="sql-tab-content">` +
        '<pre style="white-space: pre-wrap;">' + escapeHtml(data.raw_sql) + '</pre>' +
        '</div>';
    html += `<div id="sql-tab-text" class="sql-tab-content" style="display:none;">` +
        renderTable(data.text_table.fields, data.text_table.rows, 'text-table') + '</div>';
    html += `<div id="sql-tab-image" class="sql-tab-content" style="display:none;">` +
        renderTable(data.image_table.fields, data.image_table.rows, 'image-table') + '</div>';
    container.innerHTML = html;
    showSqlTab('raw');
};

window.showSqlTab = function(tab) {
    document.getElementById('sql-tab-raw').style.display = (tab === 'raw') ? '' : 'none';
    document.getElementById('sql-tab-text').style.display = (tab === 'text') ? '' : 'none';
    document.getElementById('sql-tab-image').style.display = (tab === 'image') ? '' : 'none';
    // Highlight active tab
    for (const btn of document.querySelectorAll('.sql-tab-btn')) {
        btn.classList.remove('active');
        if ((tab === 'raw' && btn.textContent === 'Raw SQL') ||
            (tab === 'text' && btn.textContent === 'Text Entries Table') ||
            (tab === 'image' && btn.textContent === 'Image Entries Table')) {
            btn.classList.add('active');
        }
    }
};
