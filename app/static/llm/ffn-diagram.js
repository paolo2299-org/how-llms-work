const SVG_NS = 'http://www.w3.org/2000/svg';

const INPUT_DIM = 8;
const HIDDEN_DIM = 24;
const OUTPUT_DIM = 8;

const W = 720;
const H = 460;
const COL_INPUT_X = 110;
const COL_HIDDEN_X = 360;
const COL_OUTPUT_X = 610;
const TOP_Y = 70;
const BOTTOM_Y = 400;

// The three interpretable hidden units we annotate with leader lines.
const HIDDEN_LABELS = {
  4:  'Physical Object',
  12: 'Negation',
  18: 'Determiner',
};
const CONCEPT_IDXS = Object.keys(HIDDEN_LABELS).map(Number);

// A plausible background of (mostly uninterpreted) hidden activations, shared
// across tokens. A few mid-bright units (idx 8, 15) reinforce that not every
// active feature maps to a human-readable concept.
const HIDDEN_BG = [
  0.10, 0.15, 0.25, 0.10, 0.16, 0.20, 0.15, 0.10,
  0.45, 0.10, 0.10, 0.10, 0.16, 0.10, 0.15, 0.55,
  0.10, 0.10, 0.16, 0.10, 0.20, 0.10, 0.10, 0.30,
];

// Build a 24-unit hidden vector where the token's concept unit fires bright and
// the other two labelled units stay dim. Tokens that share a concept therefore
// share an identical hidden pattern — the point being that the layer has learned
// one detector that several different tokens trigger.
function hiddenFor(conceptIdx) {
  const h = HIDDEN_BG.slice();
  CONCEPT_IDXS.forEach(i => { h[i] = 0.16; });
  h[conceptIdx] = 0.88;
  return h;
}

// Hand-picked illustrative activations per input token. Inputs/outputs vary per
// token (they are different tokens); the hidden layer collapses same-concept
// tokens onto the same learned feature.
const TOKENS = [
  { token: 'cat',   concept: 4,
    input:  [0.70, 0.20, 0.90, 0.40, 0.10, 0.60, 0.30, 0.80],
    output: [0.40, 0.30, 0.70, 0.50, 0.20, 0.60, 0.40, 0.50] },
  { token: 'dog',   concept: 4,
    input:  [0.65, 0.25, 0.85, 0.30, 0.15, 0.55, 0.35, 0.75],
    output: [0.45, 0.25, 0.72, 0.48, 0.18, 0.62, 0.38, 0.55] },
  { token: 'the',   concept: 18,
    input:  [0.30, 0.80, 0.20, 0.75, 0.40, 0.25, 0.70, 0.20],
    output: [0.20, 0.65, 0.30, 0.25, 0.60, 0.35, 0.55, 0.30] },
  { token: 'a',     concept: 18,
    input:  [0.35, 0.75, 0.25, 0.70, 0.45, 0.20, 0.72, 0.25],
    output: [0.25, 0.62, 0.28, 0.30, 0.58, 0.32, 0.58, 0.28] },
  { token: 'not',   concept: 12,
    input:  [0.50, 0.40, 0.30, 0.20, 0.85, 0.30, 0.40, 0.60],
    output: [0.55, 0.35, 0.40, 0.30, 0.50, 0.45, 0.35, 0.40] },
  { token: 'never', concept: 12,
    input:  [0.55, 0.35, 0.35, 0.25, 0.80, 0.35, 0.45, 0.55],
    output: [0.50, 0.38, 0.42, 0.28, 0.52, 0.48, 0.38, 0.42] },
].map(t => ({ ...t, hidden: hiddenFor(t.concept) }));

function nodeY(idx, count) {
  if (count === 1) return (TOP_Y + BOTTOM_Y) / 2;
  return TOP_Y + (BOTTOM_Y - TOP_Y) * idx / (count - 1);
}

function buildSVG(doc) {
  const svg = doc.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('class', 'ffn-diagram-svg');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label',
    'Feed-forward network architecture: 8 input neurons connected to 24 hidden neurons connected to 8 output neurons, fully connected.');

  // Edges first (so they sit behind the nodes). Each edge remembers the index of
  // its destination node so its red intensity can track that node's activation —
  // edges converging on a strongly-fired node glow brightest.
  const edgesL1 = [];
  const edgesL2 = [];
  for (let i = 0; i < INPUT_DIM; i++) {
    const y1 = nodeY(i, INPUT_DIM);
    for (let j = 0; j < HIDDEN_DIM; j++) {
      const y2 = nodeY(j, HIDDEN_DIM);
      const line = doc.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', String(COL_INPUT_X));
      line.setAttribute('y1', String(y1));
      line.setAttribute('x2', String(COL_HIDDEN_X));
      line.setAttribute('y2', String(y2));
      line.setAttribute('class', 'ffn-edge ffn-edge-l1');
      svg.appendChild(line);
      edgesL1.push({ el: line, dest: j });
    }
  }
  for (let i = 0; i < HIDDEN_DIM; i++) {
    const y1 = nodeY(i, HIDDEN_DIM);
    for (let j = 0; j < OUTPUT_DIM; j++) {
      const y2 = nodeY(j, OUTPUT_DIM);
      const line = doc.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', String(COL_HIDDEN_X));
      line.setAttribute('y1', String(y1));
      line.setAttribute('x2', String(COL_OUTPUT_X));
      line.setAttribute('y2', String(y2));
      line.setAttribute('class', 'ffn-edge ffn-edge-l2');
      svg.appendChild(line);
      edgesL2.push({ el: line, dest: j });
    }
  }

  // Nodes
  function addNodes(count, x, cls, radius) {
    const nodes = [];
    for (let i = 0; i < count; i++) {
      const node = doc.createElementNS(SVG_NS, 'circle');
      node.setAttribute('cx', String(x));
      node.setAttribute('cy', String(nodeY(i, count)));
      node.setAttribute('r', String(radius));
      node.setAttribute('class', `ffn-node ${cls}`);
      node.style.setProperty('--activation', '0');
      svg.appendChild(node);
      nodes.push(node);
    }
    return nodes;
  }
  const inputNodes  = addNodes(INPUT_DIM,  COL_INPUT_X,  'ffn-node-input',  7);
  const hiddenNodes = addNodes(HIDDEN_DIM, COL_HIDDEN_X, 'ffn-node-hidden', 5);
  const outputNodes = addNodes(OUTPUT_DIM, COL_OUTPUT_X, 'ffn-node-output', 7);

  // Column headers
  function header(x, text) {
    const t = doc.createElementNS(SVG_NS, 'text');
    t.setAttribute('x', String(x));
    t.setAttribute('y', '30');
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('class', 'ffn-col-header');
    t.textContent = text;
    svg.appendChild(t);
  }
  header(COL_INPUT_X,  'Input (post-attention)');
  header(COL_HIDDEN_X, 'Hidden layer (24 units)');
  header(COL_OUTPUT_X, 'Output');

  // Non-linearity annotation under the hidden column
  const nl = doc.createElementNS(SVG_NS, 'text');
  nl.setAttribute('x', String(COL_HIDDEN_X));
  nl.setAttribute('y', String(H - 25));
  nl.setAttribute('text-anchor', 'middle');
  nl.setAttribute('class', 'ffn-nl-label');
  nl.textContent = '+ GELU non-linearity';
  svg.appendChild(nl);

  // Hidden-unit labels with leader lines to the right
  Object.entries(HIDDEN_LABELS).forEach(([idxStr, label]) => {
    const i = Number(idxStr);
    const y = nodeY(i, HIDDEN_DIM);
    const lx = COL_HIDDEN_X + 18;

    const leader = doc.createElementNS(SVG_NS, 'line');
    leader.setAttribute('x1', String(COL_HIDDEN_X + 6));
    leader.setAttribute('y1', String(y));
    leader.setAttribute('x2', String(lx));
    leader.setAttribute('y2', String(y));
    leader.setAttribute('class', 'ffn-leader');
    svg.appendChild(leader);

    const t = doc.createElementNS(SVG_NS, 'text');
    t.setAttribute('x', String(lx + 4));
    t.setAttribute('y', String(y + 4));
    t.setAttribute('class', 'ffn-hidden-label');
    t.textContent = label;
    svg.appendChild(t);
  });

  // Input token chip + arrow — makes explicit that one token's vector is fed in,
  // and that the 8 input nodes are that token's dimensions (not 8 separate tokens).
  const chipCY = (TOP_Y + BOTTOM_Y) / 2;
  const chip = doc.createElementNS(SVG_NS, 'rect');
  chip.setAttribute('x', '12');
  chip.setAttribute('y', String(chipCY - 16));
  chip.setAttribute('width', '66');
  chip.setAttribute('height', '32');
  chip.setAttribute('rx', '16');
  chip.setAttribute('class', 'ffn-token-chip');
  svg.appendChild(chip);

  const chipText = doc.createElementNS(SVG_NS, 'text');
  chipText.setAttribute('x', '45');
  chipText.setAttribute('y', String(chipCY + 5));
  chipText.setAttribute('text-anchor', 'middle');
  chipText.setAttribute('class', 'ffn-token-chip-text');
  svg.appendChild(chipText);

  const arrow = doc.createElementNS(SVG_NS, 'path');
  arrow.setAttribute('d', `M82 ${chipCY} H100 M100 ${chipCY} l-6 -4 M100 ${chipCY} l-6 4`);
  arrow.setAttribute('class', 'ffn-token-arrow');
  svg.appendChild(arrow);

  return { svg, inputNodes, hiddenNodes, outputNodes, edgesL1, edgesL2, chipText };
}

export function initFFNDiagram({ doc = document } = {}) {
  const containerEl = doc.getElementById('ffn-diagram');
  const tokenBtnsEl = doc.getElementById('ffn-token-btns');
  if (!containerEl) return null;

  const { svg, inputNodes, hiddenNodes, outputNodes, edgesL1, edgesL2, chipText } = buildSVG(doc);
  containerEl.appendChild(svg);

  let activeToken = TOKENS[0];

  function applyActivations(tokenDef) {
    activeToken = tokenDef;
    chipText.textContent = tokenDef.token;
    tokenDef.input.forEach((v, i)  => inputNodes[i].style.setProperty('--activation', String(v)));
    tokenDef.hidden.forEach((v, i) => hiddenNodes[i].style.setProperty('--activation', String(v)));
    tokenDef.output.forEach((v, i) => outputNodes[i].style.setProperty('--activation', String(v)));
    // Edge intensity tracks the activation of the node it feeds into.
    edgesL1.forEach(({ el, dest }) => el.style.setProperty('--activation', String(tokenDef.hidden[dest])));
    edgesL2.forEach(({ el, dest }) => el.style.setProperty('--activation', String(tokenDef.output[dest])));
  }
  applyActivations(activeToken);

  // Token picker buttons
  const tokenButtons = [];
  if (tokenBtnsEl) {
    TOKENS.forEach(tokenDef => {
      const btn = doc.createElement('button');
      btn.type = 'button';
      btn.textContent = tokenDef.token;
      btn.setAttribute('aria-pressed', String(tokenDef === activeToken));
      btn.addEventListener('click', () => {
        applyActivations(tokenDef);
        tokenButtons.forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
        play();
      });
      tokenBtnsEl.appendChild(btn);
      tokenButtons.push(btn);
    });
  }

  const PHASES = ['svg-phase-input', 'svg-phase-edges1', 'svg-phase-hidden', 'svg-phase-edges2', 'svg-phase-output'];
  let timers = [];

  function resetPhases() {
    PHASES.forEach(p => svg.classList.remove(p));
  }

  function play() {
    timers.forEach(clearTimeout);
    timers = [];
    resetPhases();

    timers.push(setTimeout(() => svg.classList.add('svg-phase-input'),   100));
    timers.push(setTimeout(() => svg.classList.add('svg-phase-edges1'),  500));
    timers.push(setTimeout(() => svg.classList.add('svg-phase-hidden'),  850));
    timers.push(setTimeout(() => svg.classList.add('svg-phase-edges2'), 1450));
    timers.push(setTimeout(() => svg.classList.add('svg-phase-output'), 1800));
  }

  // Auto-play once when the diagram first scrolls into view.
  if (typeof IntersectionObserver !== 'undefined') {
    const observer = new IntersectionObserver((entries, obs) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          setTimeout(play, 300);
          obs.unobserve(entry.target);
        }
      }
    }, { threshold: 0.4 });
    observer.observe(containerEl);
  } else {
    setTimeout(play, 600);
  }

  return { play };
}
