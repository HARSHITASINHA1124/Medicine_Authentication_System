# Medicine Authentication Raspberry Pi Display

This is a deliberately minimal Raspberry Pi monitor frontend.

It contains only:

1. Scan
2. Authentication output: Genuine / Counterfeit / Suspicious

It does NOT add a dashboard, history, reports, login, or other features to the Raspberry Pi display.

## Important integration note

The repository supplied for this task could not be fetched from the current execution environment, so the exact backend routes and request/response schema could not be verified. The frontend therefore keeps the API paths in `.env` rather than inventing endpoint names.

Before running:

1. Copy `.env.example` to `.env`.
2. Set `VITE_API_BASE_URL`.
3. Set `VITE_SCAN_ENDPOINT` to the existing scan route from the backend.
4. Set `VITE_RESULT_ENDPOINT` to the existing result route if the backend exposes a separate result route.
5. If the existing scan endpoint requires a request body, update the `startScan()` adapter in `src/main.jsx` to match the backend's existing schema.

The UI is otherwise ready for the Raspberry Pi monitor workflow.

## Run

npm install
npm run dev -- --host 0.0.0.0

For production:

npm run build
npm run preview -- --host 0.0.0.0
