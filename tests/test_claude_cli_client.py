import json
import unittest
from types import SimpleNamespace
from unittest import mock

from claude_cli_client import ClaudeCliClient, split_messages, to_openai_response

CLI_SUCCESS = {
    "result": "56",
    "is_error": False,
    "usage": {"input_tokens": 2, "cache_creation_input_tokens": 600, "cache_read_input_tokens": 50,
              "output_tokens": 3},
}


class SplitMessagesTest(unittest.TestCase):
    def test_single_user_turn_is_sent_verbatim(self):
        messages = [{'role': 'system', 'content': 'Be terse.'}, {'role': 'user', 'content': '7*8?'}]

        system_prompt, prompt = split_messages(messages)

        self.assertEqual((system_prompt, prompt), ('Be terse.', '7*8?'))

    def test_multi_turn_conversation_becomes_labelled_transcript(self):
        messages = [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'},
                    {'role': 'user', 'content': 'c'}]

        _, prompt = split_messages(messages)

        self.assertEqual(prompt, 'User: a\n\nAssistant: b\n\nUser: c')


class ToOpenAiResponseTest(unittest.TestCase):
    def test_prompt_tokens_include_cached_input(self):
        response = to_openai_response(CLI_SUCCESS)

        self.assertEqual((response.usage.prompt_tokens, response.usage.completion_tokens), (652, 3))


class ClaudeCliClientTest(unittest.TestCase):
    def make_client(self):
        return ClaudeCliClient(executable="claude")

    @mock.patch("claude_cli_client.subprocess.run")
    def test_gpt_model_name_is_mapped_to_claude_alias(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(CLI_SUCCESS), stderr="")

        self.make_client().chat.completions.create(model='gpt-4-turbo', messages=[{'role': 'user', 'content': 'x'}])

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "sonnet")

    @mock.patch("claude_cli_client.subprocess.run")
    def test_cli_error_result_raises(self, run):
        failure = {**CLI_SUCCESS, "is_error": True, "result": "rate limited"}
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(failure), stderr="")

        with self.assertRaises(RuntimeError):
            self.make_client().chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': 'x'}])


if __name__ == '__main__':
    unittest.main()
