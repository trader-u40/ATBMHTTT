import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from oracle import JsdomOracle, evaluate_one, parser_changed  # noqa: E402


class OracleTests(unittest.TestCase):
    def test_script_alert_executes(self):
        result = evaluate_one("<script>alert(1)</script>")
        self.assertTrue(result.executed)
        self.assertTrue(result.parser_changed)

    def test_div_is_pasini_true_but_not_xss(self):
        payload = "<div>hello</div>"
        self.assertTrue(parser_changed(payload))
        result = evaluate_one(payload)
        self.assertFalse(result.executed)

    def test_encoded_query_executes_after_decode(self):
        payload = "http://x.test/search.php?q=%3cscript%3ealert(1)%3c/script%3e"
        result = evaluate_one(payload)
        self.assertTrue(result.executed)

    def test_title_hook(self):
        result = evaluate_one("<script>document.title='XSS_ORACLE'</script>")
        self.assertTrue(result.executed)

    def test_img_onerror_executes(self):
        result = evaluate_one("<img src=x onerror=alert(1)>")
        self.assertTrue(result.executed)

    def test_javascript_href_executes(self):
        result = evaluate_one('<a href="javascript:alert(1)">x</a>')
        self.assertTrue(result.executed)

    def test_benign_javascript_url_does_not_execute(self):
        for payload in ('<a href="javascript:void(0)">x</a>', '<a href="javascript:;">x</a>'):
            self.assertFalse(evaluate_one(payload).executed, payload)

    def test_inline_event_handlers_execute(self):
        for payload in (
            "<div onmouseover=alert(1)>x</div>",
            "<input autofocus onfocus=alert(1)>",
            "<svg onload=alert(1)>",
            "<body onload=alert(1)>",
        ):
            self.assertTrue(evaluate_one(payload).executed, payload)

    def test_inline_handler_without_sink_is_not_xss(self):
        for payload in ('<div onclick="var x=1">x</div>', '<img src=x onerror="var x=1">'):
            self.assertFalse(evaluate_one(payload).executed, payload)

    def test_infinite_loop_times_out(self):
        with JsdomOracle(timeout_s=2.0) as oracle:
            result = oracle.evaluate("<script>while(1){}</script>")
            self.assertFalse(result.executed)
            self.assertTrue(result.timeout)
            self.assertTrue(oracle.evaluate("<script>alert(1)</script>").executed)


if __name__ == "__main__":
    unittest.main()
