from risk_engine.portfolio import Portfolio
from risk_engine.report import build_risk_report


def test_report_files_created(tmp_path):
    result = build_risk_report(Portfolio(), out_dir=tmp_path)
    md = tmp_path / "risk_report.md"
    assert md.exists()
    assert result["markdown"] == str(md)
    assert len(result["screenshots"]) == 5
    for png in result["screenshots"]:
        assert png.endswith(".png")


def test_report_markdown_content(tmp_path):
    build_risk_report(Portfolio(), out_dir=tmp_path)
    text = (tmp_path / "risk_report.md").read_text(encoding="utf-8")
    for section in ["# Risk Report", "## Portfolio", "## Value-at-Risk",
                    "## Backtest", "## Stress scenarios",
                    "## Cash-Flow-at-Risk", "## Charts",
                    "Kupiec", "market_crash"]:
        assert section in text


def test_report_rerunnable(tmp_path):
    build_risk_report(Portfolio(), out_dir=tmp_path)
    build_risk_report(Portfolio(), out_dir=tmp_path)  # idempotent overwrite
    assert (tmp_path / "risk_report.md").exists()
