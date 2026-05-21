const HEADS = [
  {
    id: 'pronoun',
    label: 'Pronoun Resolution',
    description: 'Tracks which noun a pronoun like "it" refers to.',
    defaultToken: 9,
  },
  {
    id: 'syntax',
    label: 'Syntax',
    description: 'Tracks grammatical relationships, like verb → subject.',
    defaultToken: 4,
  },
  {
    id: 'local',
    label: 'Local Attention',
    description: 'Mostly attends to nearby tokens, regardless of meaning.',
    defaultToken: 9,
  },
];

//          0      1         2      3      4      5     6      7            8           9     10     11     12
const SENTENCE = ['The', 'trophy', 'did', 'not', 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too', 'large.'];

const DEFAULT_HEAD = 'pronoun';
const DEFAULT_TOKEN = 9;

// Fully curated weights for all 39 (head × token) pairs.
// Each row is the attention weight vector when that token is the query.
// Query token weight is always 0 (it does not attend to itself).
// Local-head weights are a pre-computed Gaussian (σ = 1.5) normalised per row.
const WEIGHTS = {
  pronoun: {
    //    The    trophy  did    not    fit    in     the    suitcase because  it     was    too    large.
    0:  [0,     0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    1:  [0.08,  0,      0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    2:  [0.08,  0.08,   0,     0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    3:  [0.08,  0.08,   0.08,  0,     0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    4:  [0.08,  0.08,   0.08,  0.08,  0,     0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    5:  [0.08,  0.08,   0.08,  0.08,  0.08,  0,     0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    6:  [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0,     0.08,   0.08,   0.08,  0.08,  0.08,  0.08],
    7:  [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0,      0.08,   0.08,  0.08,  0.08,  0.08],
    8:  [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0,      0.08,  0.08,  0.08,  0.08],
    9:  [0.02,  0.72,   0.02,  0.02,  0.04,  0.01,  0.01,  0.10,   0.03,   0,     0.02,  0.01,  0.01],
    10: [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0,     0.08,  0.08],
    11: [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0,     0.08],
    12: [0.08,  0.08,   0.08,  0.08,  0.08,  0.08,  0.08,  0.08,   0.08,   0.08,  0.08,  0.08,  0   ],
  },
  syntax: {
    //    The    trophy  did    not    fit    in     the    suitcase because  it     was    too    large.
    0:  [0,     0.82,   0.04,  0.02,  0.04,  0.01,  0.01,  0.03,   0.01,   0.01,  0.01,  0.01,  0   ],
    1:  [0.05,  0,      0.15,  0.03,  0.60,  0.03,  0.01,  0.05,   0.02,   0.02,  0.02,  0.01,  0.01],
    2:  [0.02,  0.05,   0,     0.08,  0.72,  0.02,  0.01,  0.02,   0.01,   0.03,  0.01,  0.01,  0.01],
    3:  [0.02,  0.04,   0.18,  0,     0.64,  0.02,  0.01,  0.02,   0.01,   0.01,  0.01,  0.01,  0.01],
    4:  [0.05,  0.60,   0.10,  0.05,  0,     0.02,  0.01,  0.07,   0.04,   0.03,  0.02,  0.01,  0.01],
    5:  [0.02,  0.04,   0.02,  0.01,  0.25,  0,     0.02,  0.55,   0.02,   0.02,  0.02,  0.01,  0.01],
    6:  [0.01,  0.03,   0.01,  0.01,  0.03,  0.02,  0,     0.82,   0.02,   0.01,  0.01,  0.01,  0.01],
    7:  [0.02,  0.03,   0.02,  0.01,  0.12,  0.38,  0.28,  0,      0.05,   0.03,  0.02,  0.01,  0.01],
    8:  [0.01,  0.03,   0.04,  0.02,  0.35,  0.02,  0.01,  0.04,   0,      0.04,  0.35,  0.04,  0.05],
    9:  [0.02,  0.06,   0.02,  0.01,  0.05,  0.01,  0.01,  0.04,   0.02,   0,     0.68,  0.04,  0.04],
    10: [0.01,  0.03,   0.02,  0.01,  0.04,  0.01,  0.01,  0.03,   0.03,   0.40,  0,     0.08,  0.33],
    11: [0.01,  0.02,   0.01,  0.01,  0.02,  0.01,  0.01,  0.02,   0.02,   0.03,  0.05,  0,     0.79],
    12: [0.01,  0.03,   0.01,  0.01,  0.03,  0.01,  0.01,  0.02,   0.02,   0.15,  0.38,  0.28,  0   ],
  },
  local: {
    // Pre-computed Gaussian (σ = 1.5), normalised so the nearest neighbour = 1.00.
    // d=1 → 1.00, d=2 → 0.51, d=3 → 0.17, d=4 → 0.04, d≥5 → 0.00
    //    The    trophy  did    not    fit    in     the    suitcase because  it     was    too    large.
    0:  [0,     1.00,   0.51,  0.17,  0.04,  0,     0,     0,      0,      0,     0,     0,     0   ],
    1:  [1.00,  0,      1.00,  0.51,  0.17,  0.04,  0,     0,      0,      0,     0,     0,     0   ],
    2:  [0.51,  1.00,   0,     1.00,  0.51,  0.17,  0.04,  0,      0,      0,     0,     0,     0   ],
    3:  [0.17,  0.51,   1.00,  0,     1.00,  0.51,  0.17,  0.04,   0,      0,     0,     0,     0   ],
    4:  [0.04,  0.17,   0.51,  1.00,  0,     1.00,  0.51,  0.17,   0.04,   0,     0,     0,     0   ],
    5:  [0,     0.04,   0.17,  0.51,  1.00,  0,     1.00,  0.51,   0.17,   0.04,  0,     0,     0   ],
    6:  [0,     0,      0.04,  0.17,  0.51,  1.00,  0,     1.00,   0.51,   0.17,  0.04,  0,     0   ],
    7:  [0,     0,      0,     0.04,  0.17,  0.51,  1.00,  0,      1.00,   0.51,  0.17,  0.04,  0   ],
    8:  [0,     0,      0,     0,     0.04,  0.17,  0.51,  1.00,   0,      1.00,  0.51,  0.17,  0.04],
    9:  [0,     0,      0,     0,     0,     0.04,  0.17,  0.51,   1.00,   0,     1.00,  0.51,  0.17],
    10: [0,     0,      0,     0,     0,     0,     0.04,  0.17,   0.51,   1.00,  0,     1.00,  0.51],
    11: [0,     0,      0,     0,     0,     0,     0,     0.04,   0.17,   0.51,  1.00,  0,     1.00],
    12: [0,     0,      0,     0,     0,     0,     0,     0,      0.04,   0.17,  0.51,  1.00,  0   ],
  },
};

const EXPLANATIONS = {
  'pronoun:0':  '"The" is a determiner, not a pronoun. The pronoun-resolution head has no target here — attention is spread evenly and low across the sentence.',
  'pronoun:1':  '"trophy" is a noun and the eventual referent of "it", but not a pronoun itself. This head only activates strongly when the query is an actual pronoun.',
  'pronoun:2':  '"did" is an auxiliary verb. The pronoun-resolution head has nothing to resolve here — no pronoun, no antecedent relationship.',
  'pronoun:3':  '"not" is a negation particle. No pronoun-antecedent relationship applies here.',
  'pronoun:4':  '"fit" is the main verb. The pronoun-resolution head finds no signal in verbs — attention is flat.',
  'pronoun:5':  '"in" is a preposition. No pronoun relationship to resolve here.',
  'pronoun:6':  '"the" is a determiner. The pronoun-resolution head is idle for determiners.',
  'pronoun:7':  '"suitcase" is a noun and one of the candidate antecedents for "it". But this head only activates when the query is actually a pronoun — try selecting "it" to see the contrast.',
  'pronoun:8':  '"because" is a conjunction. The pronoun-resolution head has no signal for conjunctions.',
  'pronoun:9':  '"it" refers to the trophy. The pronoun-resolution head looks back for a noun that could be "too large" to fit inside the suitcase. Since the trophy is the object being placed, it is the likely referent.',
  'pronoun:10': '"was" is a verb. The pronoun-resolution head has no signal for verbs.',
  'pronoun:11': '"too" is a degree adverb. The pronoun-resolution head finds nothing here.',
  'pronoun:12': '"large." is an adjective. Notice the contrast with selecting "it" — that is the only token in this sentence where the pronoun head activates strongly.',

  'syntax:0':  '"The" is an article introducing "trophy". The syntax head connects determiners to the nouns they modify.',
  'syntax:1':  '"trophy" is the grammatical subject of the verb phrase "did not fit". The syntax head points it toward the main verb "fit".',
  'syntax:2':  '"did" is an auxiliary verb supporting "fit". The syntax head links auxiliaries to their main verb.',
  'syntax:3':  '"not" is a negation particle modifying the verb phrase. The syntax head connects it to "fit", the verb it negates.',
  'syntax:4':  '"fit" is the main verb. The syntax head identifies "trophy" as its grammatical subject — the entity that did or did not fit.',
  'syntax:5':  '"in" is a preposition. The syntax head connects it to its object "suitcase" and to "fit", the verb this prepositional phrase modifies.',
  'syntax:6':  '"the" (second article) introduces "suitcase". The syntax head links it to that noun.',
  'syntax:7':  '"suitcase" is the object of the preposition "in". The syntax head links it back to "in" and the article "the" that precedes it.',
  'syntax:8':  '"because" is a subordinating conjunction. The syntax head attends to both verbs it bridges — "fit" in the main clause and "was" in the subordinate clause.',
  'syntax:9':  '"it" is the subject of the subordinate clause "it was too large". The syntax head links it to "was", the verb it governs.',
  'syntax:10': '"was" is the copular verb of the subordinate clause. The syntax head connects it to its subject "it" and its predicate adjective "large".',
  'syntax:11': '"too" is a degree modifier intensifying "large". The syntax head connects degree adverbs to the words they modify.',
  'syntax:12': '"large." is the predicate adjective. The syntax head connects it to "was" (the copula), "too" (its degree modifier), and distantly to "it" (the subject being described as large).',

  'local:0':  'At the sentence start, "The" only has neighbours to its right. The local head highlights "trophy" (one step away) most strongly, fading out quickly.',
  'local:1':  '"trophy" attends equally to its immediate neighbours "The" and "did" — one step on each side. The local head does not look further than it needs to.',
  'local:2':  '"did" attends to "trophy" and "not" on either side. Position, not grammar or meaning, drives this pattern.',
  'local:3':  '"not" attends to "did" and "fit" on either side. The local head treats all equidistant tokens identically.',
  'local:4':  '"fit" highlights "not" and "in" — its immediate neighbours — equally. The local head does not know or care that "trophy" is the grammatical subject.',
  'local:5':  '"in" attends to "fit" and "the" on either side. A preposition and a determiner light up equally just because they are close.',
  'local:6':  '"the" (second) attends to "in" and "suitcase". The local head does not recognise their grammatical link — it only sees distance.',
  'local:7':  '"suitcase" attends to "the" before it and "because" after it equally. The local head is blind to word meaning or grammatical role.',
  'local:8':  '"because" attends to "suitcase" and "it" — the nearest tokens on each side. The local head does not treat conjunctions specially.',
  'local:9':  '"it" attends to "because" and "was" — its immediate neighbours. Compare with the Pronoun Resolution head: the same token shows a completely different pattern depending on what the head is looking for.',
  'local:10': '"was" attends to "it" and "too" equally. The local head has no concept of subject-verb agreement.',
  'local:11': '"too" attends to "was" and "large." on either side. These happen to be grammatically related, but the local head highlights them purely because they are adjacent.',
  'local:12': 'At the sentence end, "large." only has neighbours to its left. The local head highlights "too" (one step away) most strongly, then "was", fading quickly.',
};

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
    const weights = WEIGHTS[state.headId][state.tokenIdx];

    headBtnsEl.querySelectorAll('button').forEach(btn => {
      btn.setAttribute('aria-pressed', String(btn.dataset.headId === state.headId));
    });

    sentenceEl.querySelectorAll('.attn-token').forEach((el, i) => {
      el.style.setProperty('--attn-weight', (weights[i] ?? 0).toFixed(3));
      el.classList.toggle('attn-query', i === state.tokenIdx);
    });

    explanationEl.textContent = EXPLANATIONS[`${state.headId}:${state.tokenIdx}`] ?? '';

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
      state.tokenIdx = head.defaultToken;
      render();
    });
    headBtnsEl.appendChild(btn);
  });

  render();

  return { state, render };
}
