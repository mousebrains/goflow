"""Render the live conversation JSONL to a Markdown script.

Reads the Claude Code session log (one JSON object per line) and produces a
single Markdown document organized turn-by-turn: each human prompt becomes a
new section, followed by Claude's visible text, tool calls (with truncated
output), and any images. Internal "thinking" blocks are noted but kept
empty — the JSONL only stores their cryptographic signature, not the text.

Re-runnable: overwrites the output file with the current state. Run after
each conversation turn to keep the script up-to-date.

Run:
    python presentation/render_session.py
    python presentation/render_session.py --max-tool-lines 60
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime


# ---------- defaults ----------
DEFAULT_LOG = ('/Users/pat/.claude/projects/-Users-pat-tpw-goflow/'
               '179d24e2-b51f-4c31-84c6-c76952b720c0.jsonl')
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'session_script.md')
SKIP_TYPES = {'permission-mode', 'file-history-snapshot', 'last-prompt',
              'queue-operation', 'system', 'attachment'}


SELF_RENDER_MARKER = 'GOFLOW session script — for video reproduction'
RE_TURN_HEADING = re.compile(r'^## Turn \d+ — \d{4}-\d{2}-\d{2}T', re.MULTILINE)


def truncate(text, max_lines):
    """Trim long text blocks while keeping head + tail context."""
    if text is None:
        return ''
    n_self = (text.count(SELF_RENDER_MARKER) +
              len(RE_TURN_HEADING.findall(text)))
    if n_self > 0:
        # A tool result that contains a copy of this rendered script (e.g.
        # `head/cat/sed session_script.md`). Leaving it in produces confusing
        # nested turn headings; elide instead.
        return f'[excerpt of session_script.md elided ({n_self} marker(s))]'
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = max_lines // 2
    tail = max_lines - head - 1
    omitted = len(lines) - head - tail
    return '\n'.join(lines[:head] +
                    [f'... [{omitted} lines omitted] ...'] +
                    lines[-tail:])


def fmt_relative(ts, t0):
    if not ts or not t0:
        return ''
    try:
        a = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        b = datetime.fromisoformat(t0.replace('Z', '+00:00'))
        delta = a - b
        secs = int(delta.total_seconds())
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f'+{h:02d}:{m:02d}:{s:02d}'
    except Exception:
        return ''


def is_system_reminder(text):
    """User text that's actually a system reminder, not a real prompt."""
    if not text:
        return False
    return bool(re.match(r'^\s*<(system-reminder|local-command-stdout|'
                         r'command-name|user-prompt-submit-hook)', text))


def is_resume_summary(text):
    """Auto-injected context-compaction message that resumes a session."""
    if not text:
        return False
    return text.lstrip().startswith(
        'This session is being continued from a previous conversation')


def block_text(block):
    """Pull human-readable text out of a content block."""
    if not isinstance(block, dict):
        return ''
    t = block.get('type')
    if t == 'text':
        return block.get('text', '')
    if t == 'thinking':
        # The JSONL only stores the signature; thinking text is not retained.
        return ''
    if t == 'tool_use':
        return ''
    if t == 'tool_result':
        c = block.get('content')
        if isinstance(c, list):
            parts = []
            for sub in c:
                if isinstance(sub, dict) and sub.get('type') == 'text':
                    parts.append(sub.get('text', ''))
                elif isinstance(sub, dict) and sub.get('type') == 'image':
                    src = sub.get('source', {})
                    parts.append(f'[image: {src.get("media_type", "?")} attachment]')
            return '\n'.join(parts)
        return c if isinstance(c, str) else ''
    if t == 'image':
        return '[image attachment]'
    return ''


def render_tool_args(name, inp, max_arg_chars=400):
    """Compact one-line summary of a tool call's args."""
    if not isinstance(inp, dict):
        return repr(inp)[:max_arg_chars]
    if name == 'Bash':
        cmd = inp.get('command', '')
        desc = inp.get('description', '')
        if desc:
            return f'{desc!r} — `{cmd[:max_arg_chars]}`'
        return f'`{cmd[:max_arg_chars]}`'
    if name in ('Read', 'NotebookEdit'):
        path = inp.get('file_path') or inp.get('notebook_path', '')
        extras = []
        for k in ('offset', 'limit', 'pages'):
            if inp.get(k) is not None:
                extras.append(f'{k}={inp[k]}')
        return path + ('  (' + ', '.join(extras) + ')' if extras else '')
    if name == 'Edit':
        path = inp.get('file_path', '')
        old = inp.get('old_string', '')[:80].replace('\n', '\\n')
        return f'{path}\n  - replace: `{old}...`'
    if name == 'Write':
        path = inp.get('file_path', '')
        n = len((inp.get('content') or '').splitlines())
        return f'{path}  ({n} lines)'
    if name == 'Agent':
        return f'{inp.get("subagent_type", "general-purpose")}: ' \
               f'{inp.get("description", "")[:120]}'
    if name in ('TaskCreate', 'TaskUpdate', 'TaskGet', 'TaskList'):
        keys = sorted(inp.keys())
        compact = {k: (str(inp[k])[:60] + ('...' if len(str(inp[k]))>60 else ''))
                   for k in keys}
        return ', '.join(f'{k}={v}' for k, v in compact.items())
    s = json.dumps(inp, ensure_ascii=False)
    if len(s) > max_arg_chars:
        s = s[:max_arg_chars] + '...'
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log',  default=DEFAULT_LOG)
    ap.add_argument('--out',  default=DEFAULT_OUT)
    ap.add_argument('--max-tool-lines', type=int, default=40,
                    help='Lines of tool result to keep before truncation')
    ap.add_argument('--max-text-lines', type=int, default=200,
                    help='Lines of assistant text to keep before truncation')
    args = ap.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f'Log not found: {args.log}')

    rows = []
    with open(args.log) as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    rows = [r for r in rows if r.get('type') not in SKIP_TYPES]
    if not rows:
        sys.exit('No content rows in log.')
    t0 = rows[0].get('timestamp', '')

    # tool_use_id -> (name, input) so we can pair results to calls
    tool_calls_by_id = {}

    out = []
    out.append(f'# GOFLOW session script — for video reproduction\n')
    out.append(f'_Source: `{args.log}`_  ')
    out.append(f'_Generated: {datetime.now().isoformat(timespec="seconds")}_  ')
    out.append(f'_Total records: {len(rows)}, '
               f'session start: {t0}_\n')
    out.append('---\n')
    out.append('Each turn begins with **Pat** (the user) and continues with '
               '**Claude** (the assistant). Tool calls are shown as labelled '
               'blocks; long tool output is truncated. Internal extended-'
               'thinking text is not stored in the JSONL — only the visible '
               'response text remains, so reasoning would have to be '
               'reconstructed from the surrounding actions for the talk.\n')
    out.append('---\n')

    turn_n = 0
    for r in rows:
        ts = r.get('timestamp', '')
        rel = fmt_relative(ts, t0)
        t = r.get('type')
        msg = r.get('message') or {}

        if t == 'user' and r.get('userType') == 'external':
            content = msg.get('content')
            text = ''
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # User content can be a list when the message is a tool_result
                # carrier; pick the first plain text block (the actual prompt).
                for blk in content:
                    if isinstance(blk, dict) and blk.get('type') == 'text':
                        text = blk.get('text', '')
                        break
                # Also expose tool_result blocks here, so a result that came
                # in alongside a follow-up prompt shows up under the prompt.
                for blk in content:
                    if isinstance(blk, dict) and blk.get('type') == 'tool_result':
                        tid = blk.get('tool_use_id')
                        name, inp = tool_calls_by_id.get(tid, ('?', {}))
                        result_text = truncate(block_text(blk),
                                               args.max_tool_lines)
                        out.append(f'**[{name}] result**:')
                        out.append('```\n' + result_text + '\n```\n')

            if is_system_reminder(text) or not text.strip():
                continue

            if is_resume_summary(text):
                turn_n += 1
                out.append(f'\n## Turn {turn_n} — {ts} ({rel})  '
                           f'_(context-resume injection — not a Pat prompt)_')
                out.append('\n> The conversation crossed the model context limit '
                           'here; the runtime auto-injected a summary of prior '
                           'turns to resume work. The body is elided in this '
                           'render — see git history and earlier turns for what '
                           'preceded this point.\n')
                continue

            turn_n += 1
            out.append(f'\n## Turn {turn_n} — {ts} ({rel})')
            out.append(f'\n### Pat\n')
            out.append(text.strip())
            out.append('')
            continue

        if t == 'user':
            # Internal user record — these carry tool_results back to the model.
            content = msg.get('content')
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get('type') == 'tool_result':
                        tid = blk.get('tool_use_id')
                        name, inp = tool_calls_by_id.get(tid, ('?', {}))
                        result_text = truncate(block_text(blk),
                                               args.max_tool_lines)
                        out.append(f'**[{name}] result**:')
                        out.append('```\n' + result_text + '\n```\n')
            continue

        if t == 'assistant':
            content = msg.get('content', [])
            if not isinstance(content, list):
                continue
            for blk in content:
                btype = blk.get('type')
                if btype == 'thinking':
                    # marker only; no text retained in JSONL
                    out.append('> _[extended thinking — not stored in log]_\n')
                elif btype == 'text':
                    text = blk.get('text', '').rstrip()
                    if not text:
                        continue
                    out.append(f'### Claude  _{rel}_\n')
                    out.append(truncate(text, args.max_text_lines))
                    out.append('')
                elif btype == 'tool_use':
                    name = blk.get('name', '?')
                    inp = blk.get('input', {})
                    tool_calls_by_id[blk.get('id', '')] = (name, inp)
                    args_str = render_tool_args(name, inp)
                    out.append(f'**[tool: {name}]**  {args_str}')
                    out.append('')
            continue

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    md = '\n'.join(out)
    with open(args.out, 'w') as f:
        f.write(md)
    print(f'Wrote {args.out}')
    print(f'  {len(md.splitlines()):,} lines, '
          f'{os.path.getsize(args.out)/1e3:.0f} KB, '
          f'{turn_n} human turns')


if __name__ == '__main__':
    main()
