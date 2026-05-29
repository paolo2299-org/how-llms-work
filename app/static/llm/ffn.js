const NEURONS = [
  {
    id: 'det',
    label: 'Determiner',
    description: 'Fires on articles like "the", "a", "an". Identical activation on both "the"s — proof the FFN is purely token-wise after attention has mixed.',
  },
  {
    id: 'noun_concrete',
    label: 'Concrete Noun',
    description: 'Fires on words for tangible, countable things. Activates on "trophy" and "suitcase".',
  },
  {
    id: 'pronoun',
    label: 'Pronoun',
    description: 'Fires on pronouns. Note it activates from the raw token identity — compare with Physical Object, which gets to "it" a different way.',
  },
  {
    id: 'action_verb',
    label: 'Action Verb',
    description: 'Fires on verbs describing physical action. "fit" is the only strong activation here.',
  },
  {
    id: 'linking_verb',
    label: 'Linking Verb',
    description: 'Fires on copulas. Strong on "was"; barely flickers on the auxiliary "did".',
  },
  {
    id: 'negation',
    label: 'Negation',
    description: 'Fires on negators — "not", "never", "no". One of the cleanest single-token features in real LLMs.',
  },
  {
    id: 'containment',
    label: 'Containment',
    description: 'Fires on words about being inside, holding, fitting. Strong on "in", moderate on "fit", mild on "suitcase".',
  },
  {
    id: 'size',
    label: 'Size',
    description: 'Fires on words describing physical magnitude. Strong on "large.", moderate on "too".',
  },
  {
    id: 'causal',
    label: 'Causality',
    description: 'Fires on causal connectors — "because", "so", "therefore". A single bright cell here.',
  },
  {
    id: 'phys_obj',
    label: 'Physical Object',
    description: 'The interesting one. Fires on tokens that refer to a physical object — including pronouns once attention has resolved them. Strong on "trophy" and "suitcase", but also on "it", because attention pulled trophy\'s features in before the FFN ran.',
  },
];

//          0      1         2      3      4      5     6      7            8           9     10     11     12
const SENTENCE = ['The', 'trophy', 'did', 'not', 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too', 'large.'];

const DEFAULT_TOKEN = 9;

// Activations[tokenIdx][neuronId] in [0, 1]. Hand-curated to tell a clean story.
const ACTIVATIONS = {
  //    det   noun  pron  actV  linkV neg   cont  size  caus  phys
  0:  { det: 0.90 },
  1:  { noun_concrete: 0.90, phys_obj: 0.85 },
  2:  { linking_verb: 0.30 },
  3:  { negation: 0.95 },
  4:  { action_verb: 0.85, containment: 0.40 },
  5:  { containment: 0.90 },
  6:  { det: 0.90 },
  7:  { noun_concrete: 0.90, phys_obj: 0.85, containment: 0.55 },
  8:  { causal: 0.95 },
  9:  { pronoun: 0.90, phys_obj: 0.70, noun_concrete: 0.35 },
  10: { linking_verb: 0.90 },
  11: { size: 0.65 },
  12: { size: 0.90 },
};

const TOKEN_EXPLANATIONS = {
  0:  '"The" is a determiner — the Determiner neuron fires cleanly. A single feature, no surprises.',
  1:  '"trophy" trips two neurons at once: Concrete Noun (it names a category of object) and Physical Object (it refers to a tangible thing). Same vector, multiple features.',
  2:  '"did" is a bleached auxiliary. The Linking Verb neuron flickers, but no feature really activates — not every token has to.',
  3:  '"not" sets off the Negation neuron sharply. One of the cleanest single-feature tokens in the sentence.',
  4:  '"fit" fires Action Verb strongly and Containment moderately — "fit" implies fitting inside something. Two features from one verb.',
  5:  '"in" fires Containment alone. Prepositions of location are a textbook FFN feature.',
  6:  'The second "the" produces identical activations to the first — Determiner, full stop. The FFN is token-wise: same input vector in, same vector out, every time.',
  7:  '"suitcase" fires Concrete Noun and Physical Object — same as trophy — plus a touch of Containment, because a suitcase holds things. Two nouns, almost the same fingerprint, with one extra feature.',
  8:  '"because" fires the Causality neuron alone. The FFN tags the logical structure of the sentence.',
  9:  'The punchline. Pronoun fires from the raw token. But Physical Object also fires at 0.70 — almost as bright as on "trophy" itself. The FFN never saw the word "trophy" here. It saw the post-attention vector, which the Pronoun Resolution head had already enriched with trophy\'s features. Attention and FFN collaborating.',
  10: '"was" fires Linking Verb. Copular "be", exactly what that feature is tuned for.',
  11: '"too" fires Size moderately. Degree adverbs aren\'t pure size words, but they cluster nearby in feature space.',
  12: '"large." fires Size strongly. An adjective of magnitude, right in the centre of what this neuron looks for.',
};

function activation(tokenIdx, neuronId) {
  return ACTIVATIONS[tokenIdx]?.[neuronId] ?? 0;
}

export function initFFN({ doc = document } = {}) {
  const gridEl = doc.getElementById('ffn-grid');
  const explanationEl = doc.getElementById('ffn-explanation-text');

  if (!gridEl || !explanationEl) return null;

  const state = {
    tokenIdx: DEFAULT_TOKEN,
    neuronId: null,
  };

  function render() {
    gridEl.querySelectorAll('.ffn-neuron-label').forEach(el => {
      el.classList.toggle('ffn-neuron-selected', el.dataset.neuronId === state.neuronId);
    });

    gridEl.querySelectorAll('.ffn-token-header').forEach(el => {
      const idx = Number(el.dataset.tokenIdx);
      el.classList.toggle('ffn-token-selected', idx === state.tokenIdx);
    });

    gridEl.querySelectorAll('.ffn-cell').forEach(el => {
      const tokenIdx = Number(el.dataset.tokenIdx);
      const neuronId = el.dataset.neuronId;
      const value = activation(tokenIdx, neuronId);
      el.style.setProperty('--ffn-activation', value.toFixed(3));
      el.classList.toggle('ffn-cell-in-token', tokenIdx === state.tokenIdx);
      el.classList.toggle('ffn-cell-in-neuron', neuronId === state.neuronId);
    });

    if (state.neuronId) {
      const neuron = NEURONS.find(n => n.id === state.neuronId);
      explanationEl.textContent = neuron?.description ?? '';
    } else {
      explanationEl.textContent = TOKEN_EXPLANATIONS[state.tokenIdx] ?? '';
    }
  }

  // Build the grid: one row per neuron, columns = tokens.
  // First row is a header row of token labels; first column of each row is the neuron label.

  // Header row
  const headerCorner = doc.createElement('div');
  headerCorner.className = 'ffn-corner';
  gridEl.appendChild(headerCorner);

  SENTENCE.forEach((word, i) => {
    const th = doc.createElement('div');
    th.className = 'ffn-token-header';
    th.textContent = word;
    th.dataset.tokenIdx = String(i);
    th.tabIndex = 0;
    th.setAttribute('role', 'button');
    th.setAttribute('aria-label', `Token: ${word}`);
    th.addEventListener('click', () => {
      state.tokenIdx = i;
      state.neuronId = null;
      render();
    });
    th.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        state.tokenIdx = i;
        state.neuronId = null;
        render();
      }
    });
    gridEl.appendChild(th);
  });

  // Neuron rows
  NEURONS.forEach(neuron => {
    const label = doc.createElement('div');
    label.className = 'ffn-neuron-label';
    label.textContent = neuron.label;
    label.dataset.neuronId = neuron.id;
    label.tabIndex = 0;
    label.setAttribute('role', 'button');
    label.setAttribute('aria-label', `Neuron: ${neuron.label}`);
    label.addEventListener('click', () => {
      state.neuronId = neuron.id;
      render();
    });
    label.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        state.neuronId = neuron.id;
        render();
      }
    });
    gridEl.appendChild(label);

    SENTENCE.forEach((_, i) => {
      const cell = doc.createElement('div');
      cell.className = 'ffn-cell';
      cell.dataset.tokenIdx = String(i);
      cell.dataset.neuronId = neuron.id;
      cell.setAttribute('aria-label', `${neuron.label} at "${SENTENCE[i]}"`);
      gridEl.appendChild(cell);
    });
  });

  // Set grid columns CSS based on token count.
  gridEl.style.setProperty('--ffn-token-count', String(SENTENCE.length));

  render();

  return { state, render };
}
