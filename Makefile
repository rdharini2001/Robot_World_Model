.PHONY: test verify reproduce retrain full

test:
	pytest -q

verify:
	PYTHONPATH=src python scripts/verify_results.py

reproduce:
	./reproduce.sh

retrain:
	./reproduce.sh --retrain

full:
	./reproduce.sh --full
