import { initTokeniser } from './tokeniser.js';
import { initAttention } from './attention.js';
import { initFFN } from './ffn.js';
import { initFFNDiagram } from './ffn-diagram.js';

initTokeniser();
initAttention();
initFFNDiagram();
initFFN();
