install_requirements:
	@pip install --upgrade pip
	@pip install -r requirements.txt

load_env:
	@echo "Configuring environment variables..."
	@if [ ! -f .env ]; then cp .env.sample .env; fi
	direnv allow .

dev_setup: install_requirements load_env

fetch_raw_anac_data:
	@echo "Fetching raw dataset from ANAC's repo..."
	rm -rf .data/raw .data/*.csv .data/*.parquet
	@python utils/fetch_data.py

fetch_weather_data:
	@echo "Fetching weather data..."
	@python utils/weather_data.py

fetch_all: fetch_raw_anac_data fetch_weather_data

fetch_consolidated_dataset:
	@echo "Fetching consolidate dataset from project's repo..."
	mkdir -p .data
	@gdown $(CONSOLIDATED_FILE_ID) --output .data

clean_zone_identifiers:
	@echo "Removing Zone.Identifier files..."
	@find .data -name "*:Zone.Identifier" -delete
	@echo "Done."
