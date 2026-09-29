"""Offline unit tests (synthetic inputs, no network)."""
from secrag.analytics.diff import diff_paragraphs
from secrag.edgar.filings import Filing
from secrag.parsers import form345
from secrag.parsers.html import html_to_text
from secrag.parsers.sections import split_sections
from secrag.pipeline.chunking import chunk_text, context_header

FORM4 = """<SEC-DOCUMENT>
<DOCUMENT><TYPE>4<TEXT><XML>
<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2025-03-14</periodOfReport>
  <issuer><issuerCik>0000000123</issuerCik><issuerName>Acme Corp</issuerName>
    <issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000999</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>1</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2025-03-13</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1,000</value></transactionShares>
        <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes><footnote id="F1">Weighted average price $50.10 to $50.90.</footnote></footnotes>
</ownershipDocument>
</XML></TEXT></DOCUMENT>"""


def test_form4_parse_sgml_wrapped():
    p = form345.parse_form345_xml(FORM4)
    assert p["form_type"] == "4"
    assert p["issuer"] == {"name": "Acme Corp", "cik": "123", "ticker": "ACME"}
    assert p["aff10b5_one"] is True
    [e] = p["entries"]
    assert e["transaction_code"] == "S" and float(e["shares"]) == 1000
    assert e["price_per_share"] is None  # value lives in a footnote
    rows = form345.transaction_rows(p)
    assert rows[0]["owner_role"] == "CFO" and rows[0]["shares_after"] == 9000
    text = form345.narrative(p)
    assert "Doe Jane (CFO)" in text and "10b5-1" in text and "Weighted average" in text


def test_form_namespace_is_stripped():
    xml = FORM4.split("<XML>")[1].split("</XML>")[0].replace(
        "<ownershipDocument>", '<ownershipDocument xmlns="http://www.sec.gov/edgar/ownership">')
    assert form345.parse_form345_xml(xml)["issuer"]["ticker"] == "ACME"


def test_html_tables_and_hidden_ixbrl():
    html = """<html><body>
      <div style="display:none"><ix:header>hidden facts</ix:header></div>
      <p><b>Item 7. Management's Discussion and Analysis</b></p>
      <table><tr><td>Net sales</td><td>$</td><td>1,234</td><td>5</td><td>%</td></tr></table>
    </body></html>"""
    text = html_to_text(html)
    assert "hidden facts" not in text
    assert "Net sales | $1,234 | 5%" in text


def test_10k_sections_skip_toc_and_cross_refs():
    text = "\n".join([
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION " * 10,
        "Item 1. | Business | 1", "Item 1A. | Risk Factors | 5",           # TOC rows
        "PART I", "Item 1. Business", "We make widgets. " * 20,
        "See Item 1A for risks.",                                          # cross-reference
        "Item 1A. Risk Factors", "Widgets may break. " * 20,
        "PART II", "Item 7. Management's Discussion and Analysis of Financial Condition",
        "Revenue grew. " * 20,
    ])
    secs = {s.key: s for s in split_sections(text, "10-K")}
    assert list(secs) == ["cover", "item_1", "item_1a", "item_7"]
    assert "Widgets may break" in secs["item_1a"].text
    assert "See Item 1A for risks." in secs["item_1"].text


def test_10q_parts_do_not_collide():
    text = "\n".join(["cover " * 50, "PART I", "Item 1. Financial Statements", "numbers " * 30,
                      "PART II", "Item 1. Legal Proceedings", "lawsuits " * 30,
                      "Item 1A. Risk Factors", "risks " * 30])
    keys = [s.key for s in split_sections(text, "10-Q")]
    assert keys == ["cover", "part1_item1", "part2_item1", "part2_item1a"]


def test_8k_items():
    text = "cover " * 60 + "\nItem 2.02 Results of Operations and Financial Condition.\nWe earned money.\n" \
           "Item 9.01 Financial Statements and Exhibits.\n99.1 Press release"
    keys = [s.key for s in split_sections(text, "8-K")]
    assert keys == ["cover", "item_2.02", "item_9.01"]


def test_chunking_respects_size_and_header():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    chunks = chunk_text(text, size=800, overlap=100)
    assert len(chunks) > 5 and all(len(c) <= 1000 for c in chunks)
    hdr = context_header({"ticker": "ACME", "form_type": "10-K", "fiscal_year": 2025,
                          "filed_date": "2025-02-01", "section_title": "Item 1A. Risk Factors"})
    assert hdr == "[ACME | 10-K | FY2025 | filed 2025-02-01 | Item 1A. Risk Factors]"


def test_ownership_primary_url_skips_xsl_view():
    f = Filing(cik=320193, ticker="AAPL", company="Apple", form_type="4",
               accession="0001140361-26-037584", filed_date="2026-09-24", report_date="",
               primary_document="xslF345X06/form4.xml")
    assert f.primary_url.endswith("/000114036126037584/form4.xml")


def test_paragraph_diff():
    old = ["Our supply chain depends on suppliers located primarily in Asia and disruptions could harm us badly.",
           "We face intense competition in all of our markets from companies with greater resources than ours."]
    new = ["Our supply chain depends on suppliers located primarily in Asia and disruptions could harm us severely.",
           "New tariffs imposed on imported components could materially increase our costs and reduce margins."]
    added, removed, modified, unchanged = diff_paragraphs(old, new)
    assert len(modified) == 1 and added == [new[1]] and removed == [old[1]]
