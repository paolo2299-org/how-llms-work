const HEADS = [
  {
    id: 'pronoun',
    label: 'Pronoun Resolution',
    description: 'Tracks which noun a pronoun like "it" refers to.',
  },
  {
    id: 'syntax',
    label: 'Syntax',
    description: 'Tracks grammatical relationships, like verb → subject.',
  },
  {
    id: 'local',
    label: 'Local Attention',
    description: 'Mostly attends to nearby tokens, regardless of meaning.',
  },
];

//         0      1         2     3     4     5     6      7           8          9     10     11    12
const SENTENCE = ['The', 'trophy', 'did', 'not', 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too', 'large.'];

const DEFAULT_HEAD = 'pronoun';
const DEFAULT_TOKEN = 9; // 'it'

// Curated weights for specific (headId, queryTokenIndex) pairs.
// All other pairs fall back to a position-based Gaussian.
const WEIGHTS = {
  pronoun: {
    9: [0.02, 0.72, 0.02, 0.02, 0.04, 0.01, 0.01, 0.10, 0.03, 0, 0.02, 0.01, 0.01],
  },
  syntax: {
    4: [0.05, 0.60, 0.10, 0.05, 0, 0.02, 0.01, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01],
  },
};

const EXPLANATIONS = {
  'pronoun:9': '"it" refers to the trophy. The pronoun-resolution head looks back for a noun that could be "too large" to fit inside the suitcase. Since the trophy is the object being placed, it is the likely referent.',
  'syntax:4': '"fit" is the main verb. The syntax head highlights "trophy" as its grammatical subject — the entity that did or did not fit.',
  'local:9': 'In the local head, "it" mainly attends to its immediate neighbours — "because" just before and "was" just after. This head pays attention based on position, not meaning.',
  'local:4': 'In the local head, "fit" attends to the words closest to it — "not" and "in" on either side. Position, not grammar or meaning, drives the pattern.',
};

function gaussianWeights(queryIdx, length, sigma) {
  const raw = Array.from({ length }, (_, i) => {
    if (i === queryIdx) return 0;
    const d = i - queryIdx;
    return Math.exp(-(d * d) / (2 * sigma * sigma));
  });
  const max = Math.max(...raw, 0.001);
  return raw.map(w => w / max);
}

function getWeights(headId, tokenIdx) {
  const headWeights = WEIGHTS[headId];
  if (headWeights && headWeights[tokenIdx] !== undefined) {
    return headWeights[tokenIdx];
  }
  const sigma = headId === 'local' ? 1.5 : 3.0;
  return gaussianWeights(tokenIdx, SENTENCE.length, sigma);
}

function getExplanation(headId, tokenIdx) {
  return EXPLANATIONS[`${headId}:${tokenIdx}`] ?? null;
}

export function initAttention({ doc = document } = {}) {
  const headBtnsEl = doc.getElementById('attention-head-btns');
  const sentenceEl = doc.getElementById('attention-sentence');
  const explanationEl = doc.getElementById('attention-explanation-text');
  const weightsDisplay = doc.getElementById('attention-weights-display');

  if (!headBtnsEl || !sentenceEl || !explanationEl) return null;

  const state = {
    headId: DEFAULT_HEAD,
    tokenIdx: DEFAULT_TOKEN,
  };

  function render() {
    const weights = getWeights(state.headId, state.tokenIdx);
    const explanation = getExplanation(state.headId, state.tokenIdx);
    const head = HEADS.find(h => h.id === state.headId);

    headBtnsEl.querySelectorAll('button').forEach(btn => {
      btn.setAttribute('aria-pressed', String(btn.dataset.headId === state.headId));
    });

    sentenceEl.querySelectorAll('.attn-token').forEach((el, i) => {
      el.style.setProperty('--attn-weight', (weights[i] ?? 0).toFixed(3));
      el.classList.toggle('attn-query', i === state.tokenIdx);
    });

    const tokenText = SENTENCE[state.tokenIdx];
    explanationEl.textContent = explanation
      ?? `"${tokenText}" with the ${head?.label ?? ''} head. ${head?.description ?? ''} Try clicking other tokens or switching heads.`;

    if (weightsDisplay) {
      weightsDisplay.innerHTML = '';
      weights.forEach((w, i) => {
        const item = doc.createElement('span');
        item.className = 'attn-weight-item';
        const strong = doc.createElement('strong');
        strong.textContent = SENTENCE[i];
        item.appendChild(strong);
        item.appendChild(doc.createTextNode(` ${w.toFixed(2)}`));
        weightsDisplay.appendChild(item);
      });
    }
  }

  SENTENCE.forEach((word, i) => {
    const span = doc.createElement('span');
    span.className = 'attn-token';
    span.textContent = word;
    span.tabIndex = 0;
    span.setAttribute('role', 'button');
    span.setAttribute('aria-label', `Token: ${word}`);
    span.addEventListener('click', () => {
      state.tokenIdx = i;
      render();
    });
    span.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        state.tokenIdx = i;
        render();
      }
    });
    sentenceEl.appendChild(span);
  });

  HEADS.forEach(head => {
    const btn = doc.createElement('button');
    btn.className = 'outline';
    btn.textContent = head.label;
    btn.dataset.headId = head.id;
    btn.addEventListener('click', () => {
      state.headId = head.id;
      render();
    });
    headBtnsEl.appendChild(btn);
  });

  render();

  return { state, render };
}
