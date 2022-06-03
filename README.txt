<student-id>.zip                        # TODO: create this
├─ task 2
│   ├── main.py                         # Main runner for the task
│   ├── requirements.txt                # Package requirements for the venv
│   ├── explore_data.py                 # pre-processing helper file
│   ├── preprocessor.py                 # pre-processing helper file
│   ├── part1
│   │   └── predictions.csv             # Predictions for Part 1
│   ├── part2
│   │   └── predictions.csv             # Predictions for Part 2
│   ├── test.feats.csv
│   ├── train.feats.csv
│   ├── train.labels.0.csv
│   └── train.labels.1.csv
├── README.txt                          # This file
├── USERS.txt                           # Us :)
└── hackaton2022.docx                   # TODO: switch to PDF


main.py usage for the base task:
Create predictions:
main.py part1 pred --train-x=train.feats.csv --train-y=train.labels.0.csv --test-x=test.feats.csv --out=./part1/predictions.csv
main.py part2 pred --train-x=train.feats.csv --train-y=train.labels.1.csv --test-x=test.feats.csv --out=./part2/predictions.csv
Create Part 3 graphs:
main.py part3 --train-x=train.feats.csv



