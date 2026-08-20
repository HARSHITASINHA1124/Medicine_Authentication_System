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
4. Set `VITE_ML_ENDPOINT` to the ML service route. If it is omitted, the frontend falls back to `VITE_SCAN_ENDPOINT`.
5. Set `VITE_RESULT_ENDPOINT` to the existing result route if the backend exposes a separate result route.

The UI is otherwise ready for the Raspberry Pi monitor workflow.

## CSV input

The scan screen accepts a headerless CSV file with exactly 10 rows and exactly six numeric columns.
The columns must be ordered as `ch450`, `ch500`, `ch550`, `ch570`, `ch600`, `ch650`.

```csv
1000,1200,1400,1500,1300,900
1001,1201,1401,1501,1301,901
...
```

Every channel value must be numeric. After validation, the frontend sends the ML service:

```json
{
	"readings": [
		{"ch450": 1000, "ch500": 1200, "ch550": 1400, "ch570": 1500, "ch600": 1300, "ch650": 900}
	]
}
```

The `readings` array contains all 10 CSV rows in their original order.

## Run

npm install
npm run dev -- --host 0.0.0.0

For production:

npm run build
npm run preview -- --host 0.0.0.0
