(function () {
  const COMPARE_ENDPOINT = 'http://localhost:8000/compare';
  const sourceTextField = document.getElementById('sourceText');
  const fineTunedTextField = document.getElementById('fineTunedText');
  const baseTextField = document.getElementById('baseText');
  const compareButton = document.getElementById('compareButton');
  const statusChip = document.getElementById('statusChip');
  let statusResetHandle = null;

  if (!compareButton) {
    return;
  }

  compareButton.addEventListener('click', function () {
    const source = sourceTextField.value;
    if (!source.trim()) {
      fineTunedTextField.value = '';
      baseTextField.value = '';
      setStatus('idle', 'Awaiting input');
      return;
    }

    compareButton.disabled = true;
    setStatus('processing', 'Comparing models...');

    performComparison(source)
      .then(function (result) {
        fineTunedTextField.value = result.fineTuned;
        baseTextField.value = result.base;
        setStatus('success', 'Comparison complete');
      })
      .catch(function (error) {
        console.error(error);
        setStatus('error', error.message || 'Comparison failed');
      })
      .finally(function () {
        compareButton.disabled = false;
      });
  });

  function performComparison(text) {
    const payload = { text: text };

    return fetch(COMPARE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    }).then(function (response) {
      if (!response.ok) {
        const httpError = new Error('Comparison request failed (' + response.status + ')');
        httpError.status = response.status;
        throw httpError;
      }

      return response.json().then(function (data) {
        const fineTuned = data && data.fine_tuned && data.fine_tuned.redacted;
        const base = data && data.base && data.base.redacted;
        
        if (typeof fineTuned !== 'string' || typeof base !== 'string') {
          const formatError = new Error('Comparison response missing redacted text');
          formatError.status = response.status;
          throw formatError;
        }

        return {
          fineTuned: fineTuned,
          base: base
        };
      });
    });
  }

  function setStatus(state, message) {
    if (!statusChip) {
      return;
    }

    clearTimeout(statusResetHandle);
    statusChip.classList.remove('processing', 'success', 'error');

    switch (state) {
      case 'processing':
        statusChip.classList.add('processing');
        statusChip.textContent = message || 'Processing...';
        break;
      case 'success':
        statusChip.classList.add('success');
        statusChip.textContent = message || 'Done';
        statusResetHandle = window.setTimeout(resetStatus, 2600);
        break;
      case 'error':
        statusChip.classList.add('error');
        statusChip.textContent = message || 'Error';
        statusResetHandle = window.setTimeout(resetStatus, 3200);
        break;
      default:
        statusChip.textContent = message || 'Standing by';
        statusResetHandle = window.setTimeout(resetStatus, 2600);
        break;
    }
  }

  function resetStatus() {
    if (!statusChip) {
      return;
    }
    statusChip.classList.remove('processing', 'success', 'error');
    statusChip.textContent = 'Standing by';
  }
})();
