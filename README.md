# Sample Order Site

This repository contains a tiny web application implemented with only the
Python standard library. It presents an order form that, when submitted with an
email address, returns a confirmation page showing a randomly generated code.

## Running the site

1. (Optional) install dependencies – the project currently has none:

   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:

   ```bash
   python app.py
   ```
3. Open http://localhost:8000 in your browser and place a sample order.

## Testing

Run the test suite with:

```bash
pytest
```
