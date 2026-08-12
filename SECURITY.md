# Security Policy

Agent libOS is experimental security-sensitive software. Reports about a
possible authority bypass, unsafe provider boundary, secret exposure, evidence
corruption, containment failure, or vulnerable dependency are welcome.

## Supported versions

| Version | Current evidence and handling |
| --- | --- |
| Current `main` and the 1.5.x release line | Target of current checked-in validation and best-effort fixes |
| 1.4.x and older snapshots | No current validation or fix commitment; handling is decided per report |

This table records repository evidence, not an owner-approved long-term support
policy or SLA. A release-status page or CI run describes validation scope, not
a promise of future maintenance. If a flaw affects an older store or artifact
format, include that version; maintainers decide the applicable remediation
range for the specific report.

## Confidential reporting status

GitHub private vulnerability reporting is **enabled** for this repository. A
repository owner confirmed the setting before the 1.5.0 release. Signed-in
reporters should use the standard
[private-report form](https://github.com/yingqi-z20/Agent-libOS/security/advisories/new)
for vulnerability facts, exploit details, credentials, private data, or a
proof of concept.

Do not put sensitive security-report content in an issue, discussion, pull
request, commit, or other public location. If the private-report form becomes
unavailable, retain the report privately and open a
[GitHub issue](https://github.com/yingqi-z20/Agent-libOS/issues/new) containing
only a non-sensitive request for the maintainers to restore the confidential
channel. The request must not identify the affected component, version,
weakness, prerequisites, impact, or reproduction. This policy deliberately
does not invent an email address or imply that another confidential channel
exists.

Include, when available:

- affected revision/version and operating system;
- the violated security property and realistic impact;
- minimal reproduction steps or a small proof of concept;
- required capabilities, configuration, provider, and trust assumptions;
- whether an external effect, credential, or real user data was involved; and
- suggested mitigations or a patch, if you have one.

Use synthetic data and disposable providers. Do not test against systems or
accounts you do not own or have explicit authorization to assess. Remove live
tokens and personal data from attachments.

## Triage and coordinated disclosure

Repository maintainers will handle reports on a best-effort basis. They will
attempt to acknowledge, reproduce, assess scope, and coordinate a remediation.
No response or fix deadline is promised.

Keep a report private until maintainers and the reporter agree that a fix and
advisory can be published, or explicitly agree to another disclosure date.
Maintainers may prepare a GitHub security advisory and request a CVE when the
impact warrants it. Credits and disclosure wording are coordinated with the
reporter; a reporter may request anonymity.

Good-faith research that respects these instructions, minimizes harm, and
avoids privacy violations or service disruption is appreciated. This policy
does not authorize testing third-party systems and does not override applicable
law or provider terms.

The implemented trust boundaries and non-goals are documented in
[docs/threat_model.md](docs/threat_model.md) and the current validation scope in
[docs/support_matrix.md](docs/support_matrix.md).
