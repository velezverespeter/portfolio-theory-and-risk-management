# Portfolio Theory & Risk Management

A collection of independent portfolio construction and risk analysis models, each in its own subfolder with a dedicated README, dataset, and source code. Built with public market data and open-source tools for applied learning and demonstration purposes.

## Analyses

| Analysis | Description | Status |
|---|---|---|
| [`efficient-frontier-var`](./efficient-frontier-var) | Markowitz efficient frontier, historical & parametric VaR/CVaR, and Monte Carlo simulation across a 5-asset diversified portfolio | Complete |
| `factor-models` | Fama-French factor decomposition of portfolio returns | Planned |
| `stress-testing` | Historical and hypothetical scenario stress testing | Planned |

Each subfolder is self-contained: its own `README.md`, `requirements.txt`, and `src/` directory, so any analysis can be run independently without needing the others installed.

## Repository Structure

```
portfolio-theory-and-risk-management/
├── README.md                    (this file)
├── LICENSE
├── .gitignore
└── efficient-frontier-var/
    ├── README.md
    ├── requirements.txt
    ├── src/
    ├── notebooks/
    ├── data/
    └── outputs/
```

## License

MIT (see LICENSE) — applies repo-wide unless a subfolder specifies otherwise.

---

<sub>This repository contains independent academic and demonstration work using publicly available data. It does not constitute investment research, financial advice, or a recommendation to buy, hold, or sell any security.</sub>
