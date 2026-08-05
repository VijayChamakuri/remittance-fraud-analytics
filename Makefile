.PHONY: install run real dashboard clean

install:
	pip install -r requirements.txt

# One documented command: builds data and runs the whole pipeline.
run:
	python run_pipeline.py

# Run on the REAL ULB dataset (after: python src/download_data.py)
real:
	python src/download_data.py
	python run_pipeline.py --real

dashboard:
	python src/build_dashboard.py
	@echo "Open dashboard/index.html in a browser."

clean:
	rm -f data/*.parquet data/synthetic_creditcard.csv models/*.joblib models/*.parquet models/*.json
	rm -rf reports/eda/* reports/evaluation/* reports/rca/* reports/ab_test/*
