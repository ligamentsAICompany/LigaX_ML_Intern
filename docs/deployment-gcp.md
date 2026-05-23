# Google Cloud Deployment

This app runs a FastAPI backend and serves the built Vite frontend from the same container. Deploy the container to Cloud Run. Google Cloud Storage (GCS) can store uploaded files, datasets, logs, or static artifacts, but GCS by itself cannot run the FastAPI backend or Hugging Face Jobs workflow.

## Prerequisites

- Google Cloud project with billing enabled.
- `gcloud` CLI installed and authenticated.
- Required APIs enabled:

```cmd
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

- Local secrets ready, but not committed to git:
  - `HF_TOKEN`
  - `OPENAI_API_KEY` if you want OpenAI-backed repair/features enabled

## One-Time Project Setup

Set your deployment values:

```cmd
set PROJECT_ID=your-gcp-project-id
set REGION=us-central1
set SERVICE_NAME=ligax-ml-intern
set ARTIFACT_REPO=ligax-containers
set IMAGE_NAME=ligax-ml-intern

gcloud config set project %PROJECT_ID%
```

Create Secret Manager secrets from your local environment. These commands read values from your current shell and do not write them into the repo.

```cmd
echo %HF_TOKEN% | gcloud secrets create hf-token --data-file=-
echo %OPENAI_API_KEY% | gcloud secrets create openai-api-key --data-file=-
```

If a secret already exists, add a new version instead:

```cmd
echo %HF_TOKEN% | gcloud secrets versions add hf-token --data-file=-
echo %OPENAI_API_KEY% | gcloud secrets versions add openai-api-key --data-file=-
```

Grant the Cloud Run runtime service account access to those secrets. Replace `PROJECT_NUMBER` after running `gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"`.

```cmd
set PROJECT_NUMBER=your-project-number
set RUNTIME_SA=%PROJECT_NUMBER%-compute@developer.gserviceaccount.com

gcloud secrets add-iam-policy-binding hf-token --member="serviceAccount:%RUNTIME_SA%" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding openai-api-key --member="serviceAccount:%RUNTIME_SA%" --role="roles/secretmanager.secretAccessor"
```

## Deploy With Cloud Build

`cloudbuild.yaml` builds the Docker image, pushes it to Artifact Registry, and deploys it to Cloud Run.

```cmd
gcloud builds submit ^
  --config cloudbuild.yaml ^
  --substitutions _REGION=%REGION%,_SERVICE_NAME=%SERVICE_NAME%,_ARTIFACT_REPO=%ARTIFACT_REPO%,_IMAGE_NAME=%IMAGE_NAME%,_HF_TOKEN_SECRET=hf-token,_OPENAI_API_KEY_SECRET=openai-api-key
```

After the build completes, get the service URL:

```cmd
gcloud run services describe %SERVICE_NAME% --region %REGION% --format="value(status.url)"
```

## Manual Docker Deploy

If you prefer to build and deploy yourself:

```cmd
gcloud artifacts repositories create %ARTIFACT_REPO% --repository-format=docker --location=%REGION%

gcloud builds submit --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/%ARTIFACT_REPO%/%IMAGE_NAME%:latest

gcloud run deploy %SERVICE_NAME% ^
  --image %REGION%-docker.pkg.dev/%PROJECT_ID%/%ARTIFACT_REPO%/%IMAGE_NAME%:latest ^
  --region %REGION% ^
  --platform managed ^
  --allow-unauthenticated ^
  --port 8080 ^
  --update-secrets HF_TOKEN=hf-token:latest,OPENAI_API_KEY=openai-api-key:latest
```

## Optional GCS Bucket

Use GCS for storage, not for running the app. If you want a bucket for uploaded source files, generated datasets, exported reports, or static artifacts:

```cmd
set GCS_BUCKET=ligax-ml-intern-artifacts
gcloud storage buckets create gs://%GCS_BUCKET% --location=%REGION%
```

Only grant the Cloud Run service account the minimum role it needs, for example object user access:

```cmd
gcloud storage buckets add-iam-policy-binding gs://%GCS_BUCKET% --member="serviceAccount:%RUNTIME_SA%" --role="roles/storage.objectUser"
```

The current app code primarily uses Hugging Face storage for datasets/models. Add a `GCS_BUCKET` environment variable only after code paths need it:

```cmd
gcloud run services update %SERVICE_NAME% --region %REGION% --set-env-vars GCS_BUCKET=%GCS_BUCKET%
```

## Runtime Notes

- The container listens on Cloud Run's `PORT` environment variable and defaults to `8080`.
- The frontend is built into `backend/static` during the Docker build and is served by FastAPI.
- Keep `.env`, service account JSON files, and API keys out of git. Use Secret Manager for deployment secrets.
- For custom domains or separate frontend hosting, set `CORS_ALLOW_ORIGINS` and optionally `ALLOWED_HOSTS` on the Cloud Run service.
