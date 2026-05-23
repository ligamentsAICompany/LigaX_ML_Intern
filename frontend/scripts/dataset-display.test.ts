import assert from 'node:assert/strict';
import { test } from 'node:test';
import { datasetUploadSummary, suggestedDatasetCommand } from '../src/utils/datasetDisplay.ts';

test('summarizes uploaded files before the Hub dataset repo', () => {
  const summary = datasetUploadSummary({
    status: 'ready',
    repoId: 'ligaments-dev/transformers-attention-qa-bundle-28af10',
    url: 'https://huggingface.co/datasets/ligaments-dev/transformers-attention-qa-bundle-28af10',
    filename: 'train.jsonl',
    files: [
      { filename: 'Transformers.pdf', format: 'pdf', size_bytes: 123 },
      { filename: 'Attention_Notes.docx', format: 'docx', size_bytes: 456 },
      { filename: 'qa_pairs.csv', format: 'csv', size_bytes: 789 },
    ],
  });

  assert.deepEqual(summary, {
    uploadedFiles: ['Transformers.pdf', 'Attention_Notes.docx', 'qa_pairs.csv'],
    hubDataset: 'ligaments-dev/transformers-attention-qa-bundle-28af10',
  });
});

test('suggests a non-technical upload fine-tune command', () => {
  assert.equal(
    suggestedDatasetCommand(),
    'fine tune uploaded dataset',
  );
});
