//! Blockers, refusals and approvals for a capability-gated model (roadmap
//! Phase 30).
//!
//! The shape is that of an approval program: capabilities in named
//! **scopes** are blocked by default, a blocked request is answered with a
//! **refusal** before any forward pass, and an **approval** -- a signed grant
//! naming the scopes it covers and when it expires -- unblocks them for the
//! holder. Blockers can be added and removed without retraining; the model
//! can additionally be *trained* to refuse (see `refusal_documents`), so the
//! gate and the weights agree even when the gate is bypassed.
//!
//! What this is not: a certification. It is a mechanism a review program can
//! adopt; whether a given policy satisfies a given organization's criteria
//! is that organization's call, and nothing here claims otherwise.
//!
//! Grants are authenticated with HMAC-SHA256 over their canonical JSON, built
//! on the `sha2` dependency this crate already has and certified against the
//! RFC 4231 test vectors. A grant that was edited, forged with another key,
//! expired, or revoked fails verification.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::antipattern::Pattern;

/// Policy file schema version.
pub const POLICY_VERSION: u32 = 1;

// ------------------------------------------------------------------ hmac --

/// HMAC-SHA256 (RFC 2104) over `message` with `key`.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let digest = Sha256::digest(key);
        k[..32].copy_from_slice(&digest);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0u8; BLOCK];
    let mut opad = [0u8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner);
    let mut out = [0u8; 32];
    out.copy_from_slice(&outer.finalize());
    out
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(text.len() % 2 == 0, "odd-length hex");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).with_context(|| format!("bad hex at {i}")))
        .collect()
}

/// Constant-time equality, so a verifier does not leak how many bytes matched.
fn equal_ct(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ------------------------------------------------------------------- key --

/// The signing key of a policy: 32 random bytes, identified by a hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    bytes: Vec<u8>,
}

impl Key {
    /// A fresh key from the operating system's randomness.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut bytes = vec![0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        anyhow::ensure!(bytes.len() >= 16, "a key needs at least 16 bytes");
        Ok(Self { bytes })
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read key {}", path.display()))?;
        Self::from_bytes(unhex(text.trim())?)
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, format!("{}\n", hex(&self.bytes))).with_context(|| format!("write key {}", path.display()))
    }

    /// First 16 hex characters of `sha256(key)`: names the key without
    /// revealing it.
    pub fn id(&self) -> String {
        hex(&Sha256::digest(&self.bytes))[..16].to_string()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

// ------------------------------------------------------------- blockers --

/// Where a blocker's patterns are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Applies {
    Prompt,
    Output,
    Both,
}

impl Applies {
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "prompt" => Ok(Self::Prompt),
            "output" => Ok(Self::Output),
            "both" => Ok(Self::Both),
            other => anyhow::bail!("unknown target {other:?}; expected prompt | output | both"),
        }
    }

    fn covers_prompt(self) -> bool {
        matches!(self, Self::Prompt | Self::Both)
    }

    fn covers_output(self) -> bool {
        matches!(self, Self::Output | Self::Both)
    }
}

/// One capability gate: patterns (the `antipattern` syntax) whose match in a
/// prompt or an output triggers a refusal unless an approval covers `scope`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Blocker {
    pub id: String,
    /// The scope an approval must name to lift this blocker, e.g.
    /// `cyber:exploit-development`.
    pub scope: String,
    #[serde(default)]
    pub description: String,
    pub patterns: Vec<String>,
    pub applies_to: Applies,
    /// What the gate answers instead of calling the model.
    pub refusal: String,
}

struct CompiledBlocker {
    id: String,
    scope: String,
    patterns: Vec<Pattern>,
    applies_to: Applies,
}

/// Where a blocker fired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    pub blocker: String,
    pub scope: String,
    /// Byte offsets into the text.
    pub start: usize,
    pub end: usize,
}

// ------------------------------------------------------------ approvals --

/// What a grant says: who, which scopes, until when.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Approval {
    pub id: String,
    pub scopes: Vec<String>,
    pub issued_unix: u64,
    pub expires_unix: u64,
    #[serde(default)]
    pub note: String,
}

impl Approval {
    /// Canonical bytes: the JSON of the approval with sorted keys, which is
    /// what `serde_json` emits for a struct, so the same approval always
    /// signs the same bytes.
    pub fn canonical(&self) -> anyhow::Result<Vec<u8>> {
        serde_json::to_vec(self).context("serialize approval")
    }

    pub fn covers(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope || s == "*")
    }
}

/// An approval plus its signature and the id of the key that made it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub approval: Approval,
    pub key_id: String,
    /// Hex HMAC-SHA256 of the approval's canonical bytes.
    pub signature: String,
}

impl Grant {
    pub fn issue(key: &Key, approval: Approval) -> anyhow::Result<Self> {
        anyhow::ensure!(!approval.scopes.is_empty(), "a grant needs at least one scope");
        anyhow::ensure!(approval.expires_unix > approval.issued_unix, "a grant must expire after it is issued");
        let signature = hex(&hmac_sha256(key.bytes(), &approval.canonical()?));
        Ok(Self { approval, key_id: key.id(), signature })
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read grant {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse grant {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize grant")?;
        std::fs::write(path, format!("{text}\n")).with_context(|| format!("write grant {}", path.display()))
    }

    /// Signature, key, expiry and revocation, in that order; the first
    /// failure names itself.
    pub fn verify(&self, key: &Key, policy: &Policy, now_unix: u64) -> anyhow::Result<()> {
        anyhow::ensure!(self.key_id == key.id(), "grant {} was signed by key {}, not {}", self.approval.id, self.key_id, key.id());
        let expected = hmac_sha256(key.bytes(), &self.approval.canonical()?);
        let given = unhex(&self.signature).context("grant signature")?;
        anyhow::ensure!(equal_ct(&expected, &given), "grant {} has a bad signature: edited or forged", self.approval.id);
        anyhow::ensure!(now_unix < self.approval.expires_unix, "grant {} expired at {}", self.approval.id, self.approval.expires_unix);
        anyhow::ensure!(!policy.revoked.contains(&self.approval.id), "grant {} was revoked", self.approval.id);
        Ok(())
    }
}

// --------------------------------------------------------------- policy --

/// The gate's configuration: blockers and the revocation list, tied to a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub version: u32,
    /// Id of the key whose grants this policy accepts.
    pub key_id: String,
    pub blockers: Vec<Blocker>,
    #[serde(default)]
    pub revoked: Vec<String>,
}

/// What the gate decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// Proceed; `approved` lists the scopes a grant lifted on the way.
    Allow { approved: Vec<String> },
    /// Answer with `refusal`; the model is not called.
    Refuse { blocker: String, scope: String, refusal: String },
}

impl Decision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }
}

impl Policy {
    pub fn new(key: &Key) -> Self {
        Self { version: POLICY_VERSION, key_id: key.id(), blockers: Vec::new(), revoked: Vec::new() }
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read policy {}", path.display()))?;
        let policy: Self = serde_json::from_str(&text).with_context(|| format!("parse policy {}", path.display()))?;
        anyhow::ensure!(policy.version == POLICY_VERSION, "policy version {} is not {POLICY_VERSION}", policy.version);
        policy.validate()?;
        Ok(policy)
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        self.validate()?;
        let text = serde_json::to_string_pretty(self).context("serialize policy")?;
        std::fs::write(path, format!("{text}\n")).with_context(|| format!("write policy {}", path.display()))
    }

    /// Every blocker has a unique id, a scope, at least one pattern, and every
    /// pattern parses and consumes something.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut ids = std::collections::HashSet::new();
        for b in &self.blockers {
            anyhow::ensure!(ids.insert(b.id.clone()), "blocker id {:?} is used twice", b.id);
            anyhow::ensure!(!b.scope.is_empty(), "blocker {:?} has no scope", b.id);
            anyhow::ensure!(!b.patterns.is_empty(), "blocker {:?} has no patterns", b.id);
            anyhow::ensure!(!b.refusal.is_empty(), "blocker {:?} has no refusal text", b.id);
            for p in &b.patterns {
                let compiled = Pattern::parse(p).with_context(|| format!("blocker {:?} pattern {p:?}", b.id))?;
                anyhow::ensure!(compiled.consumes(), "blocker {:?} pattern {p:?} matches the empty string", b.id);
            }
        }
        Ok(())
    }

    pub fn add_blocker(&mut self, blocker: Blocker) -> anyhow::Result<()> {
        anyhow::ensure!(!self.blockers.iter().any(|b| b.id == blocker.id), "blocker {:?} already exists", blocker.id);
        self.blockers.push(blocker);
        self.validate()
    }

    pub fn remove_blocker(&mut self, id: &str) -> anyhow::Result<Blocker> {
        let idx = self
            .blockers
            .iter()
            .position(|b| b.id == id)
            .with_context(|| format!("no blocker {id:?}"))?;
        Ok(self.blockers.remove(idx))
    }

    pub fn revoke(&mut self, grant_id: &str) {
        if !self.revoked.iter().any(|g| g == grant_id) {
            self.revoked.push(grant_id.to_string());
        }
    }

    pub fn scopes(&self) -> Vec<String> {
        let mut scopes: Vec<String> = self.blockers.iter().map(|b| b.scope.clone()).collect();
        scopes.sort();
        scopes.dedup();
        scopes
    }

    fn compiled(&self) -> Vec<CompiledBlocker> {
        self.blockers
            .iter()
            .map(|b| CompiledBlocker {
                id: b.id.clone(),
                scope: b.scope.clone(),
                patterns: b.patterns.iter().filter_map(|p| Pattern::parse(p).ok()).collect(),
                applies_to: b.applies_to,
            })
            .collect()
    }

    /// Every blocker that fires on `text`, for the given side.
    pub fn hits(&self, text: &str, output: bool) -> Vec<Hit> {
        let tokens = crate::antipattern::text_tokens(text);
        let mut hits = Vec::new();
        for b in self.compiled() {
            let applies = if output { b.applies_to.covers_output() } else { b.applies_to.covers_prompt() };
            if !applies {
                continue;
            }
            for p in &b.patterns {
                if let Some((start, end)) = p.find(&tokens) {
                    hits.push(Hit { blocker: b.id.clone(), scope: b.scope.clone(), start, end });
                    break;
                }
            }
        }
        hits
    }

    /// The scopes the verified grants lift right now.
    pub fn approved_scopes(&self, key: &Key, grants: &[Grant], now_unix: u64) -> Vec<String> {
        let mut scopes = Vec::new();
        for g in grants {
            if g.verify(key, self, now_unix).is_ok() {
                for s in &g.approval.scopes {
                    if !scopes.contains(s) {
                        scopes.push(s.clone());
                    }
                }
            }
        }
        scopes
    }

    fn decide(&self, hits: &[Hit], approved: &[String]) -> Decision {
        let mut lifted = Vec::new();
        for hit in hits {
            if approved.iter().any(|s| s == &hit.scope || s == "*") {
                if !lifted.contains(&hit.scope) {
                    lifted.push(hit.scope.clone());
                }
                continue;
            }
            let refusal = self
                .blockers
                .iter()
                .find(|b| b.id == hit.blocker)
                .map(|b| b.refusal.clone())
                .unwrap_or_default();
            return Decision::Refuse { blocker: hit.blocker.clone(), scope: hit.scope.clone(), refusal };
        }
        Decision::Allow { approved: lifted }
    }

    /// Gate a prompt: refuse on the first blocker no approval lifts.
    pub fn decide_prompt(&self, prompt: &str, approved: &[String]) -> Decision {
        self.decide(&self.hits(prompt, false), approved)
    }

    /// Gate an output the model already produced.
    pub fn decide_output(&self, output: &str, approved: &[String]) -> Decision {
        self.decide(&self.hits(output, true), approved)
    }
}

/// The marker prepended to an approved prompt, so a model trained with it
/// (see [`refusal_documents`]) knows the request is sanctioned.
pub fn approval_marker(scopes: &[String]) -> String {
    let mut sorted = scopes.to_vec();
    sorted.sort();
    format!("[approved:{}] ", sorted.join(","))
}

/// What the gate did with one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateOutcome {
    pub prompt_decision: Decision,
    pub output_decision: Option<Decision>,
    /// The prompt actually sent (with an approval marker when one applied).
    pub prompt_sent: Option<String>,
    pub text: String,
    pub model_called: bool,
}

/// Run `generate` behind the policy: refuse before any model call when a
/// blocker fires without approval, prepend the marker when one does, and
/// replace an output that trips an output blocker with its refusal.
pub fn gated_generate<F: FnMut(&str) -> String>(
    policy: &Policy,
    approved: &[String],
    prompt: &str,
    mut generate: F,
) -> GateOutcome {
    let prompt_decision = policy.decide_prompt(prompt, approved);
    if let Decision::Refuse { refusal, .. } = &prompt_decision {
        return GateOutcome {
            text: refusal.clone(),
            prompt_decision,
            output_decision: None,
            prompt_sent: None,
            model_called: false,
        };
    }
    let lifted = match &prompt_decision {
        Decision::Allow { approved } => approved.clone(),
        Decision::Refuse { .. } => unreachable!(),
    };
    let sent = if lifted.is_empty() { prompt.to_string() } else { format!("{}{prompt}", approval_marker(&lifted)) };
    let raw = generate(&sent);
    let output_decision = policy.decide_output(&raw, approved);
    let text = match &output_decision {
        Decision::Allow { .. } => raw,
        Decision::Refuse { refusal, .. } => refusal.clone(),
    };
    GateOutcome { prompt_decision, output_decision: Some(output_decision), prompt_sent: Some(sent), text, model_called: true }
}

/// Training documents that teach the refusal (roadmap 30.3): for every prompt
/// a blocker fires on, `prompt + refusal`; for every prompt with a supplied
/// answer, `marker + prompt + answer` -- so the model learns to refuse the
/// bare request and to comply with the approved one. Prompts no blocker
/// fires on are returned with their answers unchanged when given.
pub fn refusal_documents(policy: &Policy, prompts: &[String], answers: Option<&[String]>) -> Vec<String> {
    let mut docs = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let hits = policy.hits(prompt, false);
        let answer = answers.and_then(|a| a.get(i)).filter(|a| !a.is_empty());
        match hits.first() {
            Some(hit) => {
                let refusal = policy
                    .blockers
                    .iter()
                    .find(|b| b.id == hit.blocker)
                    .map(|b| b.refusal.clone())
                    .unwrap_or_default();
                docs.push(format!("{prompt}\n{refusal}"));
                if let Some(answer) = answer {
                    let scopes: Vec<String> = hits.iter().map(|h| h.scope.clone()).collect();
                    docs.push(format!("{}{prompt}\n{answer}", approval_marker(&scopes)));
                }
            }
            None => {
                if let Some(answer) = answer {
                    docs.push(format!("{prompt}\n{answer}"));
                }
            }
        }
    }
    docs
}

/// A starter policy with the cyber scopes an approval program would gate,
/// each with a conservative pattern set the operator is expected to extend.
pub fn starter(key: &Key) -> Policy {
    let mut policy = Policy::new(key);
    let refusal = "I can't help with that here. This capability is gated; an approved account can request it.";
    let blockers = [
        (
            "exploit-development",
            "cyber:exploit-development",
            "Requests to write or weaponize exploits for specific vulnerabilities.",
            vec![
                r"[Ww]rite\s+an?\s+exploit\b",
                r"[Ww]rite\s+exploits?\b",
                r"[Ww]eaponi[sz]e\b",
                r"\bexploit\s+for\s+CVE-\d{4}-\d+",
                r"\bexploit\s+against\s+CVE-\d{4}-\d+",
                r"\breverse\s+shell\s+payload\b",
            ],
        ),
        (
            "malware",
            "cyber:malware",
            "Requests to create malware, ransomware, keyloggers or persistence mechanisms.",
            vec![
                r"\b[Rr]ansomware\b",
                r"\b[Kk]eylogger\b",
                r"\bevade\s+EDR\b",
                r"\bevade\s+antivirus\b",
                r"\bevade\s+AV\b",
                r"\bmalware\s+that\b",
                r"\bmalware\s+which\b",
            ],
        ),
        (
            "credential-harvesting",
            "cyber:credential-harvesting",
            "Requests to phish, steal or crack credentials.",
            vec![
                r"\b[Pp]hishing\s+page\b",
                r"\b[Pp]hishing\s+kit\b",
                r"\b[Pp]hishing\s+email\b",
                r"\bsteal\s+passwords?\b",
                r"\bsteal\s+credentials?\b",
                r"\bsteal\s+cookies\b",
                r"\bcrack\s+the\s+password\b",
                r"\bcrack\s+the\s+hash\b",
                r"\bcrack\s+passwords?\b",
                r"\bcrack\s+hashes\b",
                r"\bcrack\s+hash\b",
            ],
        ),
    ];
    for (id, scope, description, patterns) in blockers {
        policy
            .add_blocker(Blocker {
                id: id.into(),
                scope: scope.into(),
                description: description.into(),
                patterns: patterns.into_iter().map(str::to_string).collect(),
                applies_to: Applies::Both,
                refusal: refusal.into(),
            })
            .expect("starter blockers are valid");
    }
    policy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key::from_bytes(b"0123456789abcdef0123456789abcdef".to_vec()).unwrap()
    }

    #[test]
    fn test_hmac_matches_rfc_4231() {
        let tc1 = hmac_sha256(&[0x0bu8; 20], b"Hi There");
        assert_eq!(hex(&tc1), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
        let tc2 = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(hex(&tc2), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
        assert_eq!(unhex(&hex(&tc2)).unwrap(), tc2.to_vec());
    }

    #[test]
    fn test_grants_verify_and_fail_for_the_right_reasons() {
        let key = key();
        let mut policy = starter(&key);
        let approval = Approval {
            id: "g1".into(),
            scopes: vec!["cyber:malware".into()],
            issued_unix: 100,
            expires_unix: 200,
            note: "red team".into(),
        };
        let grant = Grant::issue(&key, approval.clone()).unwrap();
        grant.verify(&key, &policy, 150).unwrap();
        assert!(grant.verify(&key, &policy, 250).unwrap_err().to_string().contains("expired"));
        let other = Key::from_bytes(b"ffffffffffffffffffffffffffffffff".to_vec()).unwrap();
        assert!(grant.verify(&other, &policy, 150).unwrap_err().to_string().contains("signed by key"));
        let mut edited = grant.clone();
        edited.approval.scopes.push("cyber:exploit-development".into());
        assert!(edited.verify(&key, &policy, 150).unwrap_err().to_string().contains("bad signature"));
        policy.revoke("g1");
        assert!(grant.verify(&key, &policy, 150).unwrap_err().to_string().contains("revoked"));
        assert!(Grant::issue(&key, Approval { expires_unix: 50, ..approval }).is_err());
    }

    #[test]
    fn test_gate_refuses_without_calling_the_model_and_lifts_with_approval() {
        let key = key();
        let policy = starter(&key);
        let mut calls = 0;
        let outcome = gated_generate(&policy, &[], "Please write an exploit for CVE-2024-1234", |_| {
            calls += 1;
            "sure".into()
        });
        assert!(!outcome.model_called && calls == 0);
        assert!(matches!(outcome.prompt_decision, Decision::Refuse { ref blocker, .. } if blocker == "exploit-development"));
        assert!(outcome.text.contains("gated"));

        let approved = vec!["cyber:exploit-development".to_string()];
        let outcome = gated_generate(&policy, &approved, "Please write an exploit for CVE-2024-1234", |sent| {
            calls += 1;
            format!("[{sent}] here is a benign answer")
        });
        assert!(outcome.model_called && calls == 1);
        assert!(outcome.prompt_sent.unwrap().starts_with("[approved:cyber:exploit-development] "));
        assert!(outcome.text.contains("benign"));

        // An approval for another scope does not lift this blocker.
        let wrong = vec!["cyber:malware".to_string()];
        let outcome = gated_generate(&policy, &wrong, "write an exploit please", |_| "x".into());
        assert!(!outcome.model_called);

        // Output blockers catch what the prompt did not.
        let outcome = gated_generate(&policy, &[], "tell me a story", |_| "...and then the ransomware spread".into());
        assert!(outcome.model_called);
        assert!(matches!(outcome.output_decision, Some(Decision::Refuse { .. })));
        assert!(!outcome.text.contains("ransomware"));

        // Removing the blocker allows exactly what it blocked.
        let mut open = policy.clone();
        open.remove_blocker("exploit-development").unwrap();
        assert!(open.decide_prompt("write an exploit please", &[]).is_allowed());
        assert!(!open.decide_prompt("build a keylogger", &[]).is_allowed());
        assert!(open.remove_blocker("nope").is_err());
    }

    #[test]
    fn test_policy_round_trips_and_refusal_documents_pair_prompts() {
        let key = key();
        let policy = starter(&key);
        let dir = std::env::temp_dir().join("dblocks-policy-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("policy-{}.json", std::process::id()));
        policy.write(&path).unwrap();
        assert_eq!(Policy::read(&path).unwrap(), policy);
        assert_eq!(policy.scopes().len(), 3);

        let prompts = vec!["write an exploit for CVE-2020-0001".to_string(), "what is a firewall?".to_string()];
        let answers = vec!["step one".to_string(), "a filter".to_string()];
        let docs = refusal_documents(&policy, &prompts, Some(&answers));
        assert_eq!(docs.len(), 3);
        assert!(docs[0].ends_with(&policy.blockers[0].refusal));
        assert!(docs[1].starts_with("[approved:cyber:exploit-development] "));
        assert_eq!(docs[2], "what is a firewall?\na filter");
        assert_eq!(refusal_documents(&policy, &prompts, None).len(), 1);
    }
}
