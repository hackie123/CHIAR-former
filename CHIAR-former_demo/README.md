# CHIAR-Former Live Routing Demo

## Folder Structure

```
chiar_demo/
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── config.py               # Copy from chiar_final/
├── model/                  # Copy entire folder from chiar_final/
│   ├── __init__.py
│   ├── chiar_former.py
│   ├── chiar_layer.py
│   ├── chiar_classifier.py
│   ├── meta_router.py
│   ├── rope.py
│   ├── router.py
│   ├── dct_mix.py
│   └── rbf_mix.py
├── data/                   # Copy entire folder from chiar_final/
│   ├── __init__.py
│   └── wikitext.py
├── utils/
│   ├── __init__.py
│   └── flop_counter.py
└── checkpoints/
    └── chiar_threshold_dct_attn_wikitext103_350M_best.pt  ← PLACE HERE
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Checkpoint file must be named exactly:
  `chiar_threshold_dct_attn_wikitext103_350M_best.pt`

- If no checkpoint is found, the app runs in demo mode using
  heuristic function-word routing (still shows the UI correctly).

- On CPU (16GB RAM): ~8-12 seconds per input for 400M model.
  The typewriter loading messages fill this wait time.

- The app works without a GPU. Inference is in fp32 on CPU.
