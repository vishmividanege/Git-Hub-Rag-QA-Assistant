document.addEventListener('DOMContentLoaded', () => {
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
