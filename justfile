set unstable
git_modified_py := `git status --porcelain | rg '\s*[ACMR?]+\s(.*\.pyi?)' -r '$1' | tr \\n " "`

[no-exit-message]
llm-format *paths="src tests":
  @black -q {{ git_modified_py + " " + paths }}

[no-exit-message]
llm-lint *paths="src tests":
  @for file in {{ git_modified_py + " " + paths }} ; do \
      ruff check --output-format concise -q "$file" || exit 1 ; \
  done

[no-exit-message]
llm-lint-fix *paths="src tests":
  @for file in {{ git_modified_py + " " + paths }} ; do \
      ruff check --output-format concise --fix -q "$file" || exit 1 ; \
  done

[no-exit-message]
llm-typecheck *paths="src tests":
  @for file in {{ git_modified_py + " " + paths }} ; do \
      mypy --no-error-summary "$file" || exit 1 ; \
  done

[no-exit-message]
llm-test:
  #!/usr/bin/env bash
  pytest -qq --exitfirst --no-header --tb=no --disable-warnings -r fE tests \
  | rg ^FAILED\|^ERROR \
  ; exit ${PIPESTATUS[0]}

[no-exit-message]
unveil-next-error *paths="src tests":
  #!/usr/bin/env bash
  set -o pipefail
  just llm-format {{ paths }} \
  && just llm-lint-fix {{ paths }} | head -1 \
  && just llm-typecheck {{ paths }} | head -1 \
  && just llm-test \
  && echo All done. No more errors to unveil.
