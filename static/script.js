document.addEventListener('DOMContentLoaded', () => {
    // Check if the page is being served over file:// protocol
    if (window.location.protocol === 'file:') {
        const errorOverlay = document.createElement('div');
        errorOverlay.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.9); color: #fff; display: flex;
            flex-direction: column; align-items: center; justify-content: center;
            z-index: 10000; text-align: center; padding: 20px; font-family: 'Inter', sans-serif;
        `;
        errorOverlay.innerHTML = `
            <h1 style="color: #ff4d4d; margin-bottom: 20px;">⚠️ Local Access Error</h1>
            <p style="font-size: 1.2rem; max-width: 600px; line-height: 1.6;">
                You are opening <strong>index.html</strong> directly from your file system.<br>
                This causes browser security issues that prevent communication with the backend.
            </p>
            <div style="background: #222; padding: 20px; border-radius: 12px; margin: 30px 0; border: 1px solid #444; width: 100%; max-width: 500px;">
                <p style="margin-top: 0; color: #aaa;">To fix this:</p>
                <div style="text-align: left; display: inline-block;">
                    <code style="display: block; margin-bottom: 10px; color: #00ff00;">1. Keep <strong>python backend.py</strong> running</code>
                    <code style="display: block; color: #00ff00;">2. Open <strong>http://localhost:8000</strong></code>
                </div>
            </div>
            <a href="http://localhost:8000" style="
                padding: 15px 40px; background: linear-gradient(135deg, #6e8efb, #a777e3);
                color: white; text-decoration: none; border-radius: 30px; 
                font-weight: 600; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">Go to Localhost</a>
        `;
        document.body.appendChild(errorOverlay);
    }

    const repoUrlInput = document.getElementById('repo-url');
    const processBtn = document.getElementById('process-btn');
    const statusContainer = document.getElementById('status-container');
    const statusText = document.getElementById('status-text');

    const chatSection = document.getElementById('chat-section');
    const queryInput = document.getElementById('query-input');
    const askBtn = document.getElementById('ask-btn');

    const responseContainer = document.getElementById('response-container');
    const answerText = document.getElementById('answer-text');
    const toggleSourcesBtn = document.getElementById('toggle-sources');
    const sourcesList = document.getElementById('sources-list');

    let currentRepoId = null;
    let statusInterval = null;

    // --- Process Repository ---
    processBtn.addEventListener('click', async () => {
        const repoUrl = repoUrlInput.value.trim();
        if (!repoUrl) {
            alert('Please enter a GitHub Repository URL.');
            return;
        }

        processBtn.disabled = true;
        statusContainer.classList.remove('hidden');
        statusText.innerText = 'Initializing Analysis...';
        chatSection.classList.add('hidden');
        responseContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_url: repoUrl })
            });

            if (!response.ok) throw new Error('Failed to start processing');

            const data = await response.json();
            currentRepoId = data.repo_id;

            // Start polling for status
            pollStatus(currentRepoId);
        } catch (error) {
            console.error('Error:', error);
            statusText.innerText = 'Error: ' + error.message;
            processBtn.disabled = false;
        }
    });

    const pollStatus = (repoId) => {
        if (statusInterval) clearInterval(statusInterval);

        statusInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${repoId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    clearInterval(statusInterval);
                    statusText.innerText = 'Intelligence Ready!';
                    processBtn.disabled = false;
                    chatSection.classList.remove('hidden');
                    chatSection.classList.add('fade-in');
                } else if (data.status.startsWith('failed')) {
                    clearInterval(statusInterval);
                    statusText.innerText = data.status;
                    processBtn.disabled = false;
                } else {
                    statusText.innerText = 'Analyzing Repository... (Indexing documents)';
                }
            } catch (error) {
                console.error('Status polling error:', error);
            }
        }, 2000);
    };

    // --- Query AI ---
    askBtn.addEventListener('click', async () => {
        const question = queryInput.value.trim();
        if (!question) {
            alert('Please enter a question.');
            return;
        }

        askBtn.disabled = true;
        askBtn.innerText = 'Thinking...';
        responseContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, repo_id: currentRepoId })
            });

            if (!response.ok) throw new Error('AI query failed');

            const data = await response.json();

            // Update UI with response
            answerText.innerText = data.answer;
            sourcesList.innerHTML = '';
            data.sources.forEach(source => {
                const pre = document.createElement('pre');
                pre.innerText = source;
                sourcesList.appendChild(pre);
            });

            responseContainer.classList.remove('hidden');
            responseContainer.classList.add('fade-in');
        } catch (error) {
            console.error('Query error:', error);
            alert('Error: ' + error.message);
        } finally {
            askBtn.disabled = false;
            askBtn.innerText = 'Ask Intelligence';
        }
    });

    // --- Toggle Sources ---
    toggleSourcesBtn.addEventListener('click', () => {
        sourcesList.classList.toggle('hidden');
        toggleSourcesBtn.innerText = sourcesList.classList.contains('hidden')
            ? 'View Grounding Sources'
            : 'Hide Grounding Sources';
    });
});
