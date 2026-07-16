// State Management
const state = {
    selectedFile: null,
    availableModels: [],
    ingestedMeetings: [],
    activeMeetingId: null
};

// DOM Elements
const navLinks = document.querySelectorAll('.nav-link');
const tabPanels = document.querySelectorAll('.tab-panel');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Transcribe Tab Elements
const dropZone = document.getElementById('drop-zone');
const audioFileInput = document.getElementById('audio-file-input');
const browseBtn = document.getElementById('browse-btn');
const processAudioBtn = document.getElementById('process-audio-btn');
const uploadProgressContainer = document.getElementById('upload-progress-container');
const progressStatusTitle = document.getElementById('progress-status-title');
const progressStatusDesc = document.getElementById('progress-status-desc');
const transcribeOutputGrid = document.getElementById('transcribe-output-grid');
const transcriptContent = document.getElementById('transcript-content');
const summaryContent = document.getElementById('summary-content');
const meetingTypeSelect = document.getElementById('meeting-type-select');
const modelSelect = document.getElementById('model-select');
const diarizeCheckbox = document.getElementById('diarize-checkbox');

// Chat Tab Elements
const meetingListContainer = document.getElementById('meeting-list-container');
const activeMeetingName = document.getElementById('active-meeting-name');
const chatModelSelect = document.getElementById('chat-model-select');
const chatMessagesArea = document.getElementById('chat-messages-area');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');

// Evaluate Tab Elements
const evalUidInput = document.getElementById('eval-uid-input');
const evalModelSelect = document.getElementById('eval-model-select');
const runEvalBtn = document.getElementById('run-eval-btn');
const evalProgressContainer = document.getElementById('eval-progress-container');
const evalProgressTitle = document.getElementById('eval-progress-title');
const evalProgressDesc = document.getElementById('eval-progress-desc');
const evalResultsContainer = document.getElementById('eval-results-container');
const metricWer = document.getElementById('metric-wer');
const metricRouge = document.getElementById('metric-rouge');
const metricBert = document.getElementById('metric-bert');
const metricFactual = document.getElementById('metric-factual');
const metricsTableBody = document.getElementById('metrics-table-body');
const reportMdContainer = document.getElementById('report-md-container');

// Toast Notification
const toastNotification = document.getElementById('toast-notification');
const toastIcon = document.getElementById('toast-icon');
const toastMessage = document.getElementById('toast-message');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    checkOllamaStatus();
    setupUploadZone();
    setupChat();
    setupEvaluation();
});

// 1. Navigation / Tabs
function setupNavigation() {
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            const targetId = link.getAttribute('href').substring(1);
            tabPanels.forEach(panel => {
                if (panel.id === targetId) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            });
        });
    });
}

// 2. Ollama Status Check
async function checkOllamaStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        if (data.ollama_running) {
            statusDot.className = 'status-indicator-dot green';
            statusText.innerText = `Ollama: Online (${data.models.length} model(s) available)`;
            state.availableModels = data.models;
            
            // Populate select dropdowns with available models if present
            updateModelDropdowns(data.models);
        } else {
            statusDot.className = 'status-indicator-dot red';
            statusText.innerText = 'Ollama: Offline (CUDA driver error?)';
            showToast('Ollama offline. Make sure it is running on CPU/Vulkan.', true);
        }
    } catch (err) {
        statusDot.className = 'status-indicator-dot red';
        statusText.innerText = 'Ollama: Service Error';
        console.error('Failed to get status:', err);
    }
}

function updateModelDropdowns(models) {
    if (models.length === 0) return;
    
    const dropdowns = [modelSelect, chatModelSelect, evalModelSelect];
    dropdowns.forEach(select => {
        if (!select) return;
        const currentValue = select.value;
        select.innerHTML = '';
        
        // Add models from Ollama (excluding embedding models)
        models.forEach(model => {
            if (model.toLowerCase().includes('embed')) return;
            const opt = document.createElement('option');
            opt.value = model;
            opt.innerText = model;
            select.appendChild(opt);
        });

        // Add cloud Gemini as option
        const geminiOpt = document.createElement('option');
        geminiOpt.value = 'gemini';
        geminiOpt.innerText = 'gemini (Cloud)';
        select.appendChild(geminiOpt);

        // Restore value if available
        if (models.includes(currentValue) || currentValue === 'gemini') {
            select.value = currentValue;
        } else {
            select.selectedIndex = 0;
        }
    });
}

// 3. Audio Upload Area
function setupUploadZone() {
    browseBtn.addEventListener('click', () => audioFileInput.click());
    
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Run pipeline button
    processAudioBtn.addEventListener('click', startTranscriptionPipeline);
}

function handleFileSelect(file) {
    state.selectedFile = file;
    dropZone.innerHTML = `
        <i class="fa-solid fa-file-audio upload-icon" style="color: var(--accent-green);"></i>
        <h3>Selected: ${file.name}</h3>
        <p class="upload-hint">Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
        <button class="btn btn-secondary" id="change-file-btn">Change File</button>
    `;
    
    // Bind change button
    document.getElementById('change-file-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadZone();
    });

    processAudioBtn.disabled = false;
}

function resetUploadZone() {
    state.selectedFile = null;
    processAudioBtn.disabled = true;
    dropZone.innerHTML = `
        <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
        <h3>Drag & drop meeting audio here</h3>
        <p class="upload-hint">Supports MP3, WAV, M4A up to 100MB</p>
        <button class="btn btn-secondary" id="browse-btn">
            <i class="fa-solid fa-folder-open"></i> Browse Files
        </button>
    `;
    // Re-bind browse btn
    document.getElementById('browse-btn').addEventListener('click', () => audioFileInput.click());
}

// 4. Transcription Pipeline
async function startTranscriptionPipeline() {
    if (!state.selectedFile) return;

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('diarize', diarizeCheckbox.checked);
    formData.append('meeting_type', meetingTypeSelect.value);
    formData.append('model_name', modelSelect.value);

    // Show Progress State
    uploadProgressContainer.classList.remove('hidden');
    transcribeOutputGrid.classList.add('hidden');
    processAudioBtn.disabled = true;
    
    progressStatusTitle.innerText = "Uploading Audio File...";
    progressStatusDesc.innerText = "Sending payload to the FastAPI backend server...";

    let statusCheckInterval = setInterval(() => {
        // Mock progression steps to show action
        if (progressStatusTitle.innerText === "Uploading Audio File...") {
            progressStatusTitle.innerText = "Transcribing Segment via AssemblyAI...";
            progressStatusDesc.innerText = "Running speaker diarization & deep learning models (approx 20-30s)...";
        } else if (progressStatusTitle.innerText === "Transcribing Segment via AssemblyAI...") {
            progressStatusTitle.innerText = "Generating Minutes of Meeting (MOM)...";
            progressStatusDesc.innerText = `Calling local model ${modelSelect.value} to compile actions & summary...`;
        }
    }, 15000);

    try {
        const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        clearInterval(statusCheckInterval);
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Failed during backend processing.');
        }

        const data = await res.json();

        // Reveal Outputs
        transcriptContent.innerText = data.transcript;
        summaryContent.innerHTML = parseMarkdown(data.summary);
        
        uploadProgressContainer.classList.add('hidden');
        transcribeOutputGrid.classList.remove('hidden');
        
        // Add to archives sidebar
        addMeetingToArchives(data.meeting_id);
        
        showToast('Transcription and MOM summary generated successfully!');
    } catch (err) {
        clearInterval(statusCheckInterval);
        uploadProgressContainer.classList.add('hidden');
        showToast(`Processing failed: ${err.message}`, true);
        console.error(err);
    } finally {
        processAudioBtn.disabled = false;
    }
}

// 5. Chat Interface (RAG)
function setupChat() {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessageToBrain();
        }
    });

    chatSendBtn.addEventListener('click', sendMessageToBrain);
}

function addMeetingToArchives(meetingId) {
    if (state.ingestedMeetings.includes(meetingId)) return;
    
    state.ingestedMeetings.push(meetingId);
    
    // Enable Chat Input
    chatInput.disabled = false;
    chatSendBtn.disabled = false;

    // Refresh Sidebar list
    const sidebar = document.getElementById('meeting-list-container');
    const emptyMsg = sidebar.querySelector('.empty-list-indicator');
    if (emptyMsg) emptyMsg.remove();

    // Create item
    const item = document.createElement('div');
    item.className = 'meeting-item';
    item.setAttribute('data-id', meetingId);
    
    // Shorten title
    const shortName = meetingId.length > 24 ? meetingId.substring(0, 24) + '...' : meetingId;
    
    item.innerHTML = `
        <h4>${shortName}</h4>
        <span>Ingested Archive</span>
    `;

    item.addEventListener('click', () => {
        document.querySelectorAll('.meeting-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        state.activeMeetingId = meetingId;
        activeMeetingName.innerText = shortName;
    });

    sidebar.appendChild(item);
}

async function sendMessageToBrain() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Append User Message
    appendMessage(text, 'user');
    chatInput.value = '';

    // Append Typing Indicator
    const typingBubble = appendMessage('Thinking...', 'bot typing');

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text,
                model_name: chatModelSelect.value,
                meeting_id: state.activeMeetingId
            })
        });

        typingBubble.remove();

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'RAG endpoint error.');
        }

        const data = await res.json();
        
        // Append response
        appendMessage(data.answer, 'bot', data.sources);
    } catch (err) {
        typingBubble.remove();
        appendMessage(`Error retrieving answer: ${err.message}`, 'bot');
        console.error(err);
    }
}

function appendMessage(text, sender, sources = []) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}-message`;
    
    let bubbleHtml = `<div class="message-bubble">${parseMarkdown(text)}</div>`;
    
    // Render sources if bot response has context
    if (sources && sources.length > 0) {
        let sourcesHtml = '<div class="sources-container">';
        sources.forEach((src, idx) => {
            sourcesHtml += `
                <div class="source-item" title="${src.snippet}">
                    <i class="fa-solid fa-file-lines"></i> [Source ${idx+1}] Meeting: ${src.meeting_id}
                </div>
            `;
        });
        sourcesHtml += '</div>';
        bubbleHtml = `<div class="message-bubble">${parseMarkdown(text)} ${sourcesHtml}</div>`;
    }

    msg.innerHTML = bubbleHtml;
    chatMessagesArea.appendChild(msg);
    chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
    
    return msg;
}

// 6. Evaluation Suite
function setupEvaluation() {
    runEvalBtn.addEventListener('click', runModelEvaluation);
}

async function runModelEvaluation() {
    const uid = evalUidInput.value.trim();
    const model = evalModelSelect.value;

    if (!uid) {
        showToast('Please enter a valid MeetingBank UID.', true);
        return;
    }

    // Toggle states
    evalProgressContainer.classList.remove('hidden');
    evalResultsContainer.classList.add('hidden');
    runEvalBtn.disabled = true;

    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, model_name: model })
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || errData.detail || 'Evaluation script crash.');
        }

        const data = await res.json();

        if (!data.success) {
            throw new Error(data.error);
        }

        // Render dashboard metrics
        const m = data.metrics;
        metricWer.innerText = m.transcription ? `${(m.transcription.wer * 100).toFixed(2)}%` : 'N/A';
        metricRouge.innerText = m.summarization ? m.summarization.rougeL.toFixed(4) : 'N/A';
        metricBert.innerText = m.summarization ? m.summarization.bertscore_f1.toFixed(4) : 'N/A';
        metricFactual.innerText = m.summarization ? m.summarization.factual_consistency.toFixed(4) : 'N/A';

        // Render metrics table
        populateMetricsTable(m);

        // Render Report MD
        reportMdContainer.innerHTML = parseMarkdown(data.report_md);

        // Reveal dashboard
        evalProgressContainer.classList.add('hidden');
        evalResultsContainer.classList.remove('hidden');
        showToast('Evaluation pipeline completed successfully!');
    } catch (err) {
        evalProgressContainer.classList.add('hidden');
        showToast(`Evaluation failed: ${err.message}`, true);
        console.error(err);
    } finally {
        runEvalBtn.disabled = false;
    }
}

function populateMetricsTable(m) {
    metricsTableBody.innerHTML = '';
    
    const rows = [];
    
    // Transcription
    if (m.transcription) {
        rows.push(
            ['Transcription (ASR)', 'Word Error Rate (WER)', `${(m.transcription.wer * 100).toFixed(2)}%`],
            ['Transcription (ASR)', 'Character Error Rate (CER)', `${(m.transcription.cer * 100).toFixed(2)}%`],
            ['Transcription (ASR)', 'Normalized WER', `${(m.transcription.normalized_wer * 100).toFixed(2)}%`],
            ['Transcription (ASR)', 'Latency to Transcribe', `${m.transcription.latency_seconds.toFixed(2)}s`]
        );
    }
    
    // Summarization
    if (m.summarization) {
        rows.push(
            ['Summarization', 'ROUGE-1 F1', m.summarization.rouge1.toFixed(4)],
            ['Summarization', 'ROUGE-2 F1', m.summarization.rouge2.toFixed(4)],
            ['Summarization', 'ROUGE-L F1', m.summarization.rougeL.toFixed(4)],
            ['Summarization', 'BLEU Score', m.summarization.bleu.toFixed(4)],
            ['Summarization', 'METEOR Score', m.summarization.meteor.toFixed(4)],
            ['Summarization', 'BERTScore F1', m.summarization.bertscore_f1.toFixed(4)],
            ['Summarization', 'Embedding Cosine Similarity', m.summarization.semantic_cosine.toFixed(4)],
            ['Summarization', 'Factual Consistency (SummaC/NLI)', m.summarization.factual_consistency.toFixed(4)],
            ['Summarization', 'Action Item Triple Precision', m.summarization.action_item_precision.toFixed(4)],
            ['Summarization', 'Action Item Triple Recall', m.summarization.action_item_recall.toFixed(4)],
            ['Summarization', 'Action Item Triple F1', m.summarization.action_item_f1.toFixed(4)],
            ['Summarization', 'Compression Ratio', m.summarization.compression_ratio.toFixed(2)]
        );
    }

    rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${r[0]}</td>
            <td>${r[1]}</td>
            <td>${r[2]}</td>
        `;
        metricsTableBody.appendChild(tr);
    });
}

// 7. Markdown Parser (Basic implementation)
function parseMarkdown(md) {
    if (!md) return "";
    let html = md;
    
    // Escape HTML tags to prevent XSS except specific allowed layouts
    html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Restore blockquotes
    html = html.replace(/^&gt;\s+(.*)$/gim, "<blockquote>$1</blockquote>");

    // Headers
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    
    // Unordered Lists
    html = html.replace(/^\-\s+(.*)$/gim, "<li>$1</li>");
    
    // Tables (Basic conversion)
    // Replace markdown rows with HTML table tags
    const lines = html.split('\n');
    let inTable = false;
    let tableHtml = '';
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (line.startsWith('|') && line.endsWith('|')) {
            if (!inTable) {
                inTable = true;
                tableHtml += '<table>';
            }
            
            // Check if it's separator row
            if (line.includes('---')) {
                continue; 
            }
            
            const cols = line.split('|').slice(1, -1);
            tableHtml += '<tr>';
            cols.forEach(col => {
                const tag = (tableHtml.includes('<tr><tr>') || tableHtml.endsWith('<table><tr>')) ? 'th' : 'td';
                // Remove strong markers if inside tag to keep clean look
                let cellVal = col.replace(/&lt;strong&gt;/g, '').replace(/&lt;\/strong&gt;/g, '').trim();
                tableHtml += `<${tag}>${cellVal}</${tag}>`;
            });
            tableHtml += '</tr>';
        } else {
            if (inTable) {
                inTable = false;
                tableHtml += '</table>';
                // Inject table in lines
                lines[i-1] = tableHtml;
                tableHtml = '';
            }
        }
    }
    
    if (inTable) {
        tableHtml += '</table>';
        lines[lines.length - 1] = tableHtml;
    }
    
    html = lines.join('\n');
    
    // Replace single line breaks with <br> unless inside tags
    const finalLines = html.split('\n').map(line => {
        const trimmed = line.trim();
        if (trimmed.startsWith('<h') || trimmed.startsWith('<li') || trimmed.startsWith('<block') || trimmed.startsWith('<table') || trimmed.startsWith('<tr') || trimmed.startsWith('<td') || trimmed.startsWith('<th')) {
            return line;
        }
        return line ? line + "<br>" : "";
    });
    
    return finalLines.join('\n');
}

// 8. Notifications / Toast
function showToast(message, isError = false) {
    toastMessage.innerText = message;
    
    if (isError) {
        toastNotification.className = 'toast error';
        toastIcon.className = 'fa-solid fa-circle-exclamation toast-icon';
    } else {
        toastNotification.className = 'toast';
        toastIcon.className = 'fa-solid fa-circle-check toast-icon';
    }
    
    toastNotification.classList.remove('hidden');
    
    setTimeout(() => {
        toastNotification.classList.add('hidden');
    }, 4000);
}

// Copy to Clipboard helper
function copyText(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    // Use innerText to avoid copied HTML tags
    const textToCopy = el.innerText;
    
    navigator.clipboard.writeText(textToCopy)
        .then(() => showToast('Copied to clipboard!'))
        .catch(err => showToast('Failed to copy to clipboard', true));
}
