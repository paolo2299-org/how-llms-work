const visual = document.querySelector('[data-ff-visual]');

if (visual) {
  const examples = {
    name: {
      input: [0.72, 0.38, 0.84, 0.53],
      hidden: [0.18, 0.31, 0.95, 0.12, 0.24, 0.46, 0.08, 0.58, 0.17, 0.37, 0.14, 0.27],
      output: [0.56, 0.82, 0.47, 0.69],
      feature: 'person’s name',
      sentence: 'Alice waved.',
    },
    past: {
      input: [0.31, 0.77, 0.43, 0.68],
      hidden: [0.22, 0.14, 0.19, 0.35, 0.28, 0.11, 0.93, 0.41, 0.26, 0.52, 0.16, 0.33],
      output: [0.68, 0.44, 0.76, 0.35],
      feature: 'past tense',
      sentence: 'They walked.',
    },
    disagreement: {
      input: [0.61, 0.29, 0.55, 0.87],
      hidden: [0.13, 0.36, 0.21, 0.49, 0.17, 0.39, 0.24, 0.18, 0.34, 0.96, 0.29, 0.44],
      output: [0.39, 0.71, 0.63, 0.84],
      feature: 'disagreement',
      sentence: 'I disagree.',
    },
  };

  const buttons = [...visual.querySelectorAll('[data-example]')];
  const inputBars = [...visual.querySelectorAll('.ff-input-value')];
  const hiddenUnits = [...visual.querySelectorAll('.ff-hidden-unit')];
  const outputBars = [...visual.querySelectorAll('.ff-output-value')];
  const featureLabel = visual.querySelector('[data-feature-label]');
  const description = visual.querySelector('#ff-visual-description');

  const setBarHeights = (bars, values) => {
    bars.forEach((bar, index) => {
      bar.style.height = `${18 + values[index] * 72}%`;
    });
  };

  const render = (key) => {
    const example = examples[key];
    if (!example) return;

    buttons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.example === key));
    });

    setBarHeights(inputBars, example.input);
    setBarHeights(outputBars, example.output);

    hiddenUnits.forEach((unit, index) => {
      const value = example.hidden[index];
      unit.style.opacity = String(0.2 + value * 0.8);
      unit.style.transform = `scale(${0.8 + value * 0.28})`;
    });

    featureLabel.textContent = example.feature;
    description.textContent =
      `For “${example.sentence}”, a four-dimensional input is projected into twelve feature responses. ` +
      `The ${example.feature} response is strongest. GELU applies a nonlinear activation before a second linear layer returns a four-dimensional output.`;
  };

  buttons.forEach((button) => {
    button.addEventListener('click', () => render(button.dataset.example));
  });
}
