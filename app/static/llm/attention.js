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

// Curated attention weights for specific (headId, queryTokenIndex) pairs.
// Unspecified pairs fall back to a position-based Gaussian.
// Sentence token indices are shown in comments for reference.
const EXAMPLES = [
  {
    id: 'pronoun-large',
    label: 'Pronoun: too large',
    //          0      1         2     3     4     5     6      7           8          9     10     11    12
    sentence: ['The', 'trophy', 'did', 'not', 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too', 'large.'],
    defaultHead: 'pronoun',
    defaultToken: 9,
    weights: {
      pronoun: {
        9: [0.02, 0.72, 0.02, 0.02, 0.04, 0.01, 0.01, 0.10, 0.03, 0, 0.02, 0.01, 0.01],
      },
      syntax: {
        4: [0.05, 0.60, 0.10, 0.05, 0, 0.02, 0.01, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01],
      },
    },
    explanations: {
      'pronoun:9': '"it" refers to the trophy. The pronoun-resolution head looks back for a noun that could be "too large" to fit inside the suitcase. Since the trophy is the object being placed, it is the likely referent. Try switching to "Pronoun: too small" to see how a single word changes the answer.',
      'syntax:4': '"fit" is the main verb. The syntax head highlights "trophy" as its grammatical subject — the entity that did or did not fit.',
    },
  },
  {
    id: 'pronoun-small',
    label: 'Pronoun: too small',
    //          0      1         2     3     4     5     6      7           8          9     10     11    12
    sentence: ['The', 'trophy', 'did', 'not', 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too', 'small.'],
    defaultHead: 'pronoun',
    defaultToken: 9,
    weights: {
      pronoun: {
        9: [0.01, 0.09, 0.01, 0.01, 0.03, 0.01, 0.01, 0.76, 0.04, 0, 0.02, 0.01, 0.01],
      },
      syntax: {
        4: [0.05, 0.60, 0.10, 0.05, 0, 0.02, 0.01, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01],
      },
    },
    explanations: {
      'pronoun:9': '"it" now refers to the suitcase. Changing "too large" to "too small" shifts the likely referent: if the container was too small, that explains why the trophy did not fit. Compare with "Pronoun: too large" — the same head, same sentence structure, but a different answer.',
      'syntax:4': '"fit" is still the main verb, and "trophy" is still its grammatical subject regardless of which noun "it" refers to.',
    },
  },
  {
    id: 'cat-sat',
    label: 'Subject and verb',
    //          0      1      2      3     4      5
    sentence: ['The', 'cat', 'sat', 'on', 'the', 'mat.'],
    defaultHead: 'syntax',
    defaultToken: 2,
    weights: {
      syntax: {
        2: [0.04, 0.78, 0, 0.04, 0.05, 0.09],
        3: [0.02, 0.05, 0.44, 0, 0.05, 0.44],
      },
      pronoun: {},
    },
    explanations: {
      'syntax:2': '"sat" is the verb. The syntax head identifies "cat" as its grammatical subject — the one performing the action of sitting.',
      'syntax:3': 'The preposition "on" connects the action to the location. The syntax head attends to "sat" (the verb it modifies) and "mat" (the object of the preposition).',
    },
  },
  {
    id: 'local-bridge',
    label: 'Local attention',
    //          0      1      2         3         4          5      6
    sentence: ['The', 'old', 'wooden', 'bridge', 'crossed', 'the', 'river.'],
    defaultHead: 'local',
    defaultToken: 3,
    weights: {
      syntax: {
        4: [0.03, 0.05, 0.08, 0.62, 0, 0.07, 0.15],
        3: [0.05, 0.35, 0.42, 0, 0.12, 0.03, 0.03],
      },
      pronoun: {},
    },
    explanations: {
      'local:3': '"bridge" in the local head pays most attention to its immediate neighbours — "wooden" just before and "crossed" just after. Some real attention heads learn this kind of positional, neighbourhood-based pattern.',
      'local:4': '"crossed" attends mainly to "bridge" (one step left) and "the" (one step right). The local head does not look far.',
      'local:0': 'At the start of the sentence, "The" can only look forward. The local head attends to "old" immediately to its right.',
      'syntax:4': 'The syntax head links "crossed" to its grammatical subject, "bridge".',
      'syntax:3': 'The syntax head connects "bridge" to its modifiers "old" and "wooden".',
    },
  },
];

function gaussianWeights(queryIdx, length, sigma) {
  const raw = Array.from({ length }, (_, i) => {
    if (i === queryIdx) return 0;
    const d = i - queryIdx;
    return Math.exp(-(d * d) / (2 * sigma * sigma));
  });
  const max = Math.max(...raw, 0.001);
  return raw.map(w => w / max);
}

function getWeights(example, headId, tokenIdx) {
  const headWeights = example.weights[headId];
  if (headWeights && headWeights[tokenIdx] !== undefined) {
    return headWeights[tokenIdx];
  }
  const sigma = headId === 'local' ? 1.5 : 3.0;
  return gaussianWeights(tokenIdx, example.sentence.length, sigma);
}

function getExplanation(example, headId, tokenIdx) {
  return example.explanations?.[`${headId}:${tokenIdx}`] ?? null;
}

export function initAttention({ doc = document } = {}) {
  const exampleBtnsEl = doc.getElementById('attention-example-btns');
  const headBtnsEl = doc.getElementById('attention-head-btns');
  const sentenceEl = doc.getElementById('attention-sentence');
  const explanationEl = doc.getElementById('attention-explanation-text');
  const weightsDisplay = doc.getElementById('attention-weights-display');

  if (!exampleBtnsEl || !headBtnsEl || !sentenceEl || !explanationEl) return null;

  const state = {
    exampleId: EXAMPLES[0].id,
    headId: EXAMPLES[0].defaultHead,
    tokenIdx: EXAMPLES[0].defaultToken,
  };

  function currentExample() {
    return EXAMPLES.find(e => e.id === state.exampleId);
  }

  function render() {
    const example = currentExample();
    const weights = getWeights(example, state.headId, state.tokenIdx);
    const explanation = getExplanation(example, state.headId, state.tokenIdx);
    const head = HEADS.find(h => h.id === state.headId);

    exampleBtnsEl.querySelectorAll('button').forEach(btn => {
      btn.setAttribute('aria-pressed', String(btn.dataset.exampleId === state.exampleId));
    });

    headBtnsEl.querySelectorAll('button').forEach(btn => {
      btn.setAttribute('aria-pressed', String(btn.dataset.headId === state.headId));
    });

    sentenceEl.querySelectorAll('.attn-token').forEach((el, i) => {
      const w = weights[i] ?? 0;
      el.style.setProperty('--attn-weight', w.toFixed(3));
      el.classList.toggle('attn-query', i === state.tokenIdx);
    });

    const tokenText = example.sentence[state.tokenIdx];
    explanationEl.textContent = explanation
      ?? `"${tokenText}" with the ${head?.label ?? ''} head. ${head?.description ?? ''} Try clicking other tokens or switching heads.`;

    if (weightsDisplay) {
      weightsDisplay.innerHTML = '';
      weights.forEach((w, i) => {
        const item = doc.createElement('span');
        item.className = 'attn-weight-item';
        const strong = doc.createElement('strong');
        strong.textContent = example.sentence[i];
        item.appendChild(strong);
        item.appendChild(doc.createTextNode(` ${w.toFixed(2)}`));
        weightsDisplay.appendChild(item);
      });
    }
  }

  function buildSentence(example) {
    sentenceEl.innerHTML = '';
    example.sentence.forEach((word, i) => {
      const span = doc.createElement('span');
      span.className = 'attn-token';
      span.textContent = word;
      span.tabIndex = 0;
      span.setAttribute('role', 'button');
      span.setAttribute('aria-label', `Token: ${word}`);
      span.dataset.tokenIdx = String(i);
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
  }

  EXAMPLES.forEach(example => {
    const btn = doc.createElement('button');
    btn.className = 'outline';
    btn.textContent = example.label;
    btn.dataset.exampleId = example.id;
    btn.addEventListener('click', () => {
      state.exampleId = example.id;
      state.headId = example.defaultHead;
      state.tokenIdx = example.defaultToken;
      buildSentence(currentExample());
      render();
    });
    exampleBtnsEl.appendChild(btn);
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

  buildSentence(currentExample());
  render();

  return { state, render };
}
