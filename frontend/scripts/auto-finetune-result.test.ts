import assert from 'node:assert/strict';
import { test } from 'node:test';
import { autoFineTuneResultFromOutput } from '../src/utils/autoFineTuneResult.ts';

test('ignores Hugging Face docs links when extracting auto fine-tune results', () => {
  const result = autoFineTuneResultFromOutput([
    'Warning: HF_XET_HIGH_PERFORMANCE can be configured at',
    'https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance',
    'Job URL: https://huggingface.co/jobs/acme/demo-job',
  ].join('\n'));

  assert.equal(result.model_repo_url, undefined);
  assert.equal(result.job_url, 'https://huggingface.co/jobs/acme/demo-job');
});

test('extracts model URL only from explicit auto fine-tune marker in logs', () => {
  const result = autoFineTuneResultFromOutput([
    'See provider docs at https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables',
    'AUTO_FINETUNE_MODEL_URL=https://huggingface.co/acme/tax-classifier-ft',
  ].join('\n'));

  assert.equal(result.model_repo_url, 'https://huggingface.co/acme/tax-classifier-ft');
});

test('sanitizes structured auto fine-tune model and job fields', () => {
  const result = autoFineTuneResultFromOutput(JSON.stringify({
    model_url: 'https://huggingface.co/acme/tax-classifier-ft',
    job_url: 'https://huggingface.co/jobs/acme/demo-job',
    eval_result: 'accuracy=0.91',
  }));

  assert.deepEqual(result, {
    model_repo_url: 'https://huggingface.co/acme/tax-classifier-ft',
    job_url: 'https://huggingface.co/jobs/acme/demo-job',
    eval_result: 'accuracy=0.91',
  });
});

test('drops structured docs URLs from model fields', () => {
  const result = autoFineTuneResultFromOutput(JSON.stringify({
    model_repo_url: 'https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance',
    job_url: 'https://huggingface.co/jobs/acme/demo-job',
  }));

  assert.equal(result.model_repo_url, undefined);
  assert.equal(result.job_url, 'https://huggingface.co/jobs/acme/demo-job');
});
