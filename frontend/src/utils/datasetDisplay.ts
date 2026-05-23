import type { DatasetUploadState } from '@/types/agent';

export interface DatasetUploadSummary {
  uploadedFiles: string[];
  hubDataset?: string;
}

export function datasetUploadSummary(dataset: DatasetUploadState): DatasetUploadSummary {
  const uploadedFiles = (dataset.files ?? [])
    .map((file) => file.filename?.trim())
    .filter((filename): filename is string => Boolean(filename));

  if (!uploadedFiles.length && dataset.filename && dataset.filename !== 'train.jsonl') {
    uploadedFiles.push(dataset.filename);
  }

  return {
    uploadedFiles,
    hubDataset: dataset.repoId?.trim() || undefined,
  };
}

export function suggestedDatasetCommand(): string {
  return 'fine tune uploaded dataset';
}
