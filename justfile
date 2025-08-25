git_modified_py := `git status --porcelain | rg '\s*[ACMR?]+\s(.*\.pyi?)' -r '$1' | tr \\n " "`
fix-error-msg := 'Now fix this and run \`just see-what-you-did-there\` after each change.'
do-commit-now := 'Well done. Now make focused commits with concise, descriptive messages.'

[no-exit-message]
llm-syntax *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  compile='python -m compileall -j4 -q'
  test "" != "{{ git_modified_py }}" && $compile {{ git_modified_py }}
  $compile {{ paths }}

[no-exit-message]
llm-format-check *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  ruff='ruff format --quiet --check'
  test "" != "{{ git_modified_py }}" && $ruff {{ git_modified_py }}
  $ruff {{ paths }}

[no-exit-message]
llm-format *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  ruff='ruff format --quiet'
  test "" != "{{ git_modified_py }}" && $ruff {{ git_modified_py }}
  $ruff {{ paths }}

[no-exit-message]
llm-lint *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  ruff='ruff check --output-format concise --quiet'
  test "" != "{{ git_modified_py }}" && $ruff {{ git_modified_py }}
  $ruff {{ paths }}

[no-exit-message]
llm-lint-fix *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  ruff='ruff check --output-format concise --quiet --fix'
  test "" != "{{ git_modified_py }}" && $ruff {{ git_modified_py }}
  $ruff {{ paths }}

[no-exit-message]
llm-deadcode *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  test "" != "{{ git_modified_py }}" && vulture {{ git_modified_py }}
  vulture {{ paths }}

[no-exit-message]
llm-bandit:
  #!/usr/bin/env bash
  set -euo pipefail
  tpl='{abspath}:{line}: {test_id}[bandit]: {severity}: {msg}'
  bandit --format custom --msg-template "$tpl" --quiet -r src

[no-exit-message]
llm-typecheck *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  mypy='mypy --no-error-summary'
  test "" != "{{ git_modified_py }}" && $mypy {{ git_modified_py }}
  $mypy {{ paths }}

[no-exit-message]
llm-test *paths="tests":
  #!/usr/bin/env bash
  set -euo pipefail
  export MCP_LOGGING__LEVEL=ERROR
  export PYTEST_ADDOPTS="--tb=short --disable-warnings -r fE"
  pytest {{ paths }} 2>&1

[no-exit-message]
llm-test-next *paths="tests":
  #!/usr/bin/env bash
  set -euo pipefail
  export MCP_LOGGING__LEVEL=ERROR
  export PYTEST_ADDOPTS="--exitfirst --no-cov --tb=no --disable-warnings -r fE"
  pytest {{ paths }} 2>&1 | sed -ne /^FAILED/p -e /^ERROR/p

[no-exit-message]
pre-commit:
  #!/usr/bin/env bash
  set -euo pipefail
  just llm-syntax | head -1 \
  && just llm-format | head -1 \
  && just llm-lint-fix | head -1 \
  && just llm-deadcode | head -1 \
  && just llm-bandit | head -1 \
  && just llm-typecheck | head -1

[no-exit-message]
see-what-you-did-there:
  #!/usr/bin/env bash
  set -euo pipefail
  err() { echo "{{ fix-error-msg }}" ; exit 1 ; }
  ok() { echo "{{ do-commit-now }}" ; exit 0 ; }
  just pre-commit || err
  ok


[no-exit-message]
unveil-next-error *paths="src tests":
  #!/usr/bin/env bash
  set -euo pipefail
  just llm-syntax {{ paths }} | head -1 \
  && just llm-format {{ paths }} | head -1 \
  && just llm-lint {{ paths }} | head -1 \
  && just llm-deadcode {{ paths }} | head -1 \
  && just llm-bandit | head -1 \
  && just llm-typecheck {{ paths }} | head -1 \
  && just llm-test-next {{ paths }} | head -1 \
  && echo All done. No more errors to unveil.

[no-exit-message]
watch:
  #!/usr/bin/env bash
  just llm-syntax \
  && just llm-format-check \
  && just llm-lint \
  && just llm-deadcode \
  && just llm-bandit \
  && just llm-typecheck
