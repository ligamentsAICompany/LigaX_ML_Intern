@echo off
setlocal

if "%PROJECT_ID%"=="" (
  echo PROJECT_ID is required.
  echo Example: set PROJECT_ID=your-gcp-project-id
  exit /b 1
)

if "%REGION%"=="" set REGION=us-central1
if "%SERVICE_NAME%"=="" set SERVICE_NAME=ligax-ml-intern
if "%ARTIFACT_REPO%"=="" set ARTIFACT_REPO=ligax-containers
if "%IMAGE_NAME%"=="" set IMAGE_NAME=ligax-ml-intern

set IMAGE=%REGION%-docker.pkg.dev/%PROJECT_ID%/%ARTIFACT_REPO%/%IMAGE_NAME%:latest

gcloud config set project %PROJECT_ID% || exit /b 1
gcloud artifacts repositories describe %ARTIFACT_REPO% --location %REGION% >nul 2>nul
if errorlevel 1 (
  gcloud artifacts repositories create %ARTIFACT_REPO% --repository-format=docker --location %REGION% || exit /b 1
)

gcloud builds submit --tag %IMAGE% . || exit /b 1
gcloud run deploy %SERVICE_NAME% --image %IMAGE% --region %REGION% --platform managed --allow-unauthenticated --port 8080 || exit /b 1

echo.
echo Deployment complete. To attach secrets, run:
echo gcloud run services update %SERVICE_NAME% --region %REGION% --update-secrets HF_TOKEN=hf-token:latest,OPENAI_API_KEY=openai-api-key:latest
