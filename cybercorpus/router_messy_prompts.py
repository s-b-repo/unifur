#!/usr/bin/env python3
"""Messy-STYLE training prompts, to close the train/eval style gap.

The clean templated train set and the messy v2 eval differ in surface form
(typos, fragments, "halp", lowercase, run-ons), and the router — a hashed
n-gram model — keys on surface form, so it generalises poorly from clean train
to messy eval. These are TRAINING prompts written in the v2 style but on FRESH
topics (not the eval's), so the model sees messy text at train time. Imported by
router_train_prompts.py, which hash-dedupes them against the clean train, v1 and
v2 before emitting, so nothing here can leak an eval prompt into training.

MESSY_SHORT[i] = (text, [(cat, weight)...], [subfield...]). One category each.
Authored by hand; deliberately NOT drawn from the term bank or either eval file.
"""

from __future__ import annotations

# ~10 per category, fresh topics, messy surface form.
MESSY_SHORT: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    # coding
    ("ugh my rust lifetimes wont line up, the compiler wants a 'static and idk why", [("coding", 1.0)], ["coding/rust"]),
    ("how do u do a hashmap in cpp again, keep forgetting the syntax", [("coding", 1.0)], ["coding/cpp"]),
    ("python list comprehension vs a for loop, is one faster or is it just style", [("coding", 1.0)], ["coding/python"]),
    ("my go program deadlocks on a channel, send on a closed one maybe??", [("coding", 1.0)], ["coding/go"]),
    ("whats the deal with == vs === in js, i always mess this up", [("coding", 1.0)], ["coding/js-ts"]),
    ("need a bash loop over files with spaces in the name, keeps breaking", [("coding", 1.0)], ["coding/shell"]),
    ("java streams, how do i group by a field and count, blanking rn", [("coding", 1.0)], ["coding/java"]),
    ("ruby each vs map, when does it matter", [("coding", 1.0)], ["coding/ruby"]),
    ("sql join is duplicating rows, did i do the on clause wrong", [("coding", 1.0)], ["coding/sql"]),
    ("kotlin null safety keeps forcing me to use !! is that bad", [("coding", 1.0)], ["coding/kotlin"]),
    # systems
    ("linux box wont boot, stuck at initramfs prompt, halp", [("systems", 1.0)], ["systems/linux"]),
    ("whats an inode and why did i run out of them with disk space free??", [("systems", 1.0)], ["systems/linux"]),
    ("my thread pool deadlocks sometimes, race condition i think", [("systems", 1.0)], ["systems/concurrency"]),
    ("how do i see what syscalls a program makes, strace?", [("systems", 1.0)], ["systems/linux"]),
    ("writing to a gpio pin on a pi, nothing happens, wrong pin number?", [("systems", 1.0)], ["systems/embedded"]),
    ("windows handle leak, how do i even find it", [("systems", 1.0)], ["systems/windows-internals"]),
    ("whats the diff between mutex and semaphore, keep mixing up", [("systems", 1.0)], ["systems/concurrency"]),
    ("my kernel module panics on load, null deref somewhere", [("systems", 1.0)], ["systems/kernel"]),
    ("rtos task never runs, priority too low maybe", [("systems", 1.0)], ["systems/rtos"]),
    ("firmware update bricked the board, is it recoverable", [("systems", 1.0)], ["systems/firmware"]),
    # networking
    ("ping works but curl times out, whats going on", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("dns resolves externally but not internal hostnames, split horizon?", [("networking", 1.0)], ["networking/dns"]),
    ("whats mtu and why do big packets get dropped sometimes", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("my https cert works in chrome but curl says self signed??", [("networking", 1.0)], ["networking/tls"]),
    ("how do i route traffic between two subnets", [("networking", 1.0)], ["networking/routing-bgp"]),
    ("wifi says connected no internet, dhcp issue?", [("networking", 1.0)], ["networking/wireless"]),
    ("whats a 3 way handshake, syn synack ack right", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("port forwarding on my router isnt working for game server", [("networking", 1.0)], ["networking/protocols"]),
    ("explain subnet mask like /24 vs /16, always confused", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("tcpdump shows retransmits everywhere, bad link?", [("networking", 1.0)], ["networking/tcp-ip"]),
    # cloud
    ("my lambda times out after 3 sec but the work takes longer, how extend", [("cloud", 1.0)], ["cloud/aws"]),
    ("k8s service has no endpoints, selector typo?", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("docker build super slow every time, not using cache?", [("cloud", 1.0)], ["cloud/docker"]),
    ("terraform plan shows changes i didnt make, drift?", [("cloud", 1.0)], ["cloud/terraform-iac"]),
    ("how do i roll back a bad deploy fast", [("cloud", 1.0)], ["cloud/ci-cd"]),
    ("gcp cloud run cold starts are killing me", [("cloud", 1.0)], ["cloud/gcp"]),
    ("whats an slo again and how is it diff from an sla", [("cloud", 1.0)], ["cloud/sre"]),
    ("my pod oomkilled, how do i set memory limits right", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("azure function wont pick up my env var", [("cloud", 1.0)], ["cloud/azure"]),
    ("ci runner out of disk mid build, how do i clean up", [("cloud", 1.0)], ["cloud/ci-cd"]),
    # data
    ("my join is slow even with an index, is it not using it??", [("data", 1.0)], ["data/databases"]),
    ("whats a deadlock in postgres and how do i avoid it", [("data", 1.0)], ["data/databases"]),
    ("kafka rebalancing constantly, consumers keep dropping", [("data", 1.0)], ["data/streaming"]),
    ("how do i backfill a huge table without locking it", [("data", 1.0)], ["data/data-engineering"]),
    ("star schema vs snowflake, which for a small warehouse", [("data", 1.0)], ["data/analytics"]),
    ("whats idempotency in a data pipeline and why care", [("data", 1.0)], ["data/data-engineering"]),
    ("my spark job spills to disk constantly, too little mem?", [("data", 1.0)], ["data/data-engineering"]),
    ("how do i dedupe 100m rows efficiently", [("data", 1.0)], ["data/databases"]),
    ("whats a cte and is it slower than a subquery", [("data", 1.0)], ["data/databases"]),
    ("eventual consistency bit me, how do i handle stale reads", [("data", 1.0)], ["data/distributed-systems"]),
    # web-mobile
    ("my api returns 200 but the body is empty, cors?", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("react useeffect runs twice in dev, is that a bug", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do i paginate infinite scroll without dupes", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("android app anr on startup, doing too much on main thread?", [("web-mobile", 1.0)], ["web-mobile/android"]),
    ("whats the right way to store a token in a web app, localstorage bad?", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("ios keyboard covers my text field, how do i scroll it up", [("web-mobile", 1.0)], ["web-mobile/ios"]),
    ("my css animation janky on mobile, how do i smooth it", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("rest vs graphql for a crud app, overkill?", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("how do i debounce a search input in react", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("my webhook fires twice sometimes, how do i dedupe", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    # ai-ml
    ("my val loss goes down then back up, overfitting or lr too high", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("whats the diff between batchnorm and layernorm, when which", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("model predicts all one class, imbalanced data?", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("how do i pick k in kmeans, elbow thing?", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("whats a good eval metric for a ranking model", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("my transformer runs out of memory, grad checkpointing?", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("tokenizer splitting my words weird, is that normal", [("ai-ml", 1.0)], ["ai-ml/nlp"]),
    ("how do i deploy a model, onnx or just a flask api", [("ai-ml", 1.0)], ["ai-ml/mlops"]),
    ("whats transfer learning, can i reuse a pretrained net", [("ai-ml", 1.0)], ["ai-ml/cv"]),
    ("my rl agent just spins in circles, reward shaping?", [("ai-ml", 1.0)], ["ai-ml/rl"]),
    # cs-theory
    ("is memoization the same as dp or different", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    ("whats the time complexity of quicksort worst case", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("when do i use a heap vs a sorted list", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("whats a dag and why does topological sort need one", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    ("explain recursion vs iteration, stack overflow risk?", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    ("whats the difference between p and np, simply", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("how does a bloom filter work, false positives?", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("writing a tokenizer for a toy language, regex or hand-rolled", [("cs-theory", 1.0)], ["cs-theory/compilers"]),
    ("whats a dfa vs nfa", [("cs-theory", 1.0)], ["cs-theory/formal-methods"]),
    ("why is binary search log n", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    # math
    ("whats the chain rule again, derivative of a composed function", [("math", 1.0)], ["math/calculus"]),
    ("how do i invert a 2x2 matrix quick", [("math", 1.0)], ["math/linear-algebra"]),
    ("whats the difference between mean median mode, keep forgetting", [("math", 1.0)], ["math/statistics"]),
    ("explain conditional probability, the P(A|B) thing", [("math", 1.0)], ["math/probability"]),
    ("whats a prime factorization and why does crypto use it", [("math", 1.0)], ["math/number-theory"]),
    ("how do i find the area under a curve", [("math", 1.0)], ["math/calculus"]),
    ("whats a dot product geometrically", [("math", 1.0)], ["math/linear-algebra"]),
    ("how many ways to arrange 5 books on a shelf", [("math", 1.0)], ["math/combinatorics"]),
    ("whats a derivative actually measuring", [("math", 1.0)], ["math/calculus"]),
    ("solve 3x - 7 = 2x + 5 pls show steps", [("math", 1.0)], ["math/algebra"]),
    # science
    ("why does bread rise, is it the yeast making gas", [("science", 1.0)], ["science/biology"]),
    ("whats the difference between an atom and a molecule", [("science", 1.0)], ["science/chemistry"]),
    ("why is the sky blue, something about scattering", [("science", 1.0)], ["science/physics"]),
    ("how do batteries actually store energy", [("science", 1.0)], ["science/chemistry"]),
    ("whats photosynthesis in simple terms", [("science", 1.0)], ["science/biology"]),
    ("why do we have seasons, is it the earths tilt", [("science", 1.0)], ["science/earth-space"]),
    ("whats gravity really, like why does mass attract", [("science", 1.0)], ["science/physics"]),
    ("how do antibiotics kill bacteria but not us", [("science", 1.0)], ["science/biology"]),
    ("whats an isotope", [("science", 1.0)], ["science/chemistry"]),
    ("why does the moon have phases", [("science", 1.0)], ["science/earth-space"]),
    # cyber-offensive
    ("got a shell but its a weird restricted one, how do i break out", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("how do i find hidden directories on a web server", [("cyber-offensive", 1.0)], ["cyber-offensive/recon"]),
    ("whats a stack canary and how does it stop overflows", [("cyber-offensive", 1.0)], ["cyber-offensive/binary-exploitation"]),
    ("found creds in a config, how do i use them to move further", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("whats the diff between bind and reverse shell again", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("nmap says filtered not closed, whats that mean", [("cyber-offensive", 1.0)], ["cyber-offensive/scanning"]),
    ("how do i test for command injection in a form", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("whats a nop sled for in an exploit", [("cyber-offensive", 1.0)], ["cyber-offensive/exploit-dev"]),
    ("how do i enum smb shares anonymously", [("cyber-offensive", 1.0)], ["cyber-offensive/enumeration"]),
    ("whats an idor, found sequential ids in a url", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    # cyber-defensive
    ("too many false positive alerts on powershell, how do i tune", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("whats the first thing to do when a box might be compromised", [("cyber-defensive", 1.0)], ["cyber-defensive/incident-response"]),
    ("how do i tell if a process is malicious from its behavior", [("cyber-defensive", 1.0)], ["cyber-defensive/malware-analysis"]),
    ("whats a good log source for detecting lateral movement", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("how do i do a memory dump and analyze it", [("cyber-defensive", 1.0)], ["cyber-defensive/forensics"]),
    ("whats mitre attack and how do i map detections to it", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("how do i baseline normal so i can spot weird", [("cyber-defensive", 1.0)], ["cyber-defensive/threat-intel"]),
    ("whats the diff between ioc and ttp", [("cyber-defensive", 1.0)], ["cyber-defensive/threat-intel"]),
    ("how do i harden ssh on a public server", [("cyber-defensive", 1.0)], ["cyber-defensive/hardening"]),
    ("reversing a sample, where do i start with ghidra", [("cyber-defensive", 1.0)], ["cyber-defensive/reverse-engineering"]),
    # cyber-appsec
    ("my fuzzer isnt finding anything, bad corpus?", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("whats the diff between a reflected and stored xss", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how do i check if user input reaches a sql query", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("sast flagged 200 things, how do i know whats real", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("found a crash while fuzzing, is it exploitable or just a bug", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("whats a good way to review auth code for holes", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how does dast find bugs without the source", [("cyber-appsec", 1.0)], ["cyber-appsec/dast"]),
    ("whats a use after free and why is it dangerous", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("how do i add a sanitizer to catch memory bugs", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("whats the right way to handle secrets in code", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    # cyber-netsec-crypto
    ("is md5 still ok for anything or totally dead", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("whats the diff between symmetric and asymmetric crypto", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how do i tell what cipher a tls connection negotiated", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/packet-analysis"]),
    ("whats a mac and how is it diff from a hash", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how do i block a port with iptables", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/firewalls"]),
    ("whats salt in password hashing for", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("ids keeps alerting on normal traffic, tune the rules?", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/ids-ips"]),
    ("how do i verify a cert chain by hand", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/pki"]),
    ("whats a replay attack and how do nonces stop it", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how do i read a pcap in wireshark to find a login", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/packet-analysis"]),
    # cyber-specialized
    ("how do i start with car hacking, the can bus thing", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats a good first step for osint on a company", [("cyber-specialized", 1.0)], ["cyber-specialized/osint"]),
    ("my s3 bucket perms, how do i check theyre not public", [("cyber-specialized", 1.0)], ["cyber-specialized/cloud-sec"]),
    ("how do i intercept an apps https traffic on android", [("cyber-specialized", 1.0)], ["cyber-specialized/mobile-sec"]),
    ("whats a pretext for a phishing test thats not too mean", [("cyber-specialized", 1.0)], ["cyber-specialized/social-engineering"]),
    ("how do routers and iot cams usually get owned", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats the deal with rogue access points at cafes", [("cyber-specialized", 1.0)], ["cyber-specialized/wireless-sec"]),
    ("how do i scope a pentest so i dont go out of bounds", [("cyber-specialized", 1.0)], ["cyber-specialized/methodology"]),
    ("whats involved in responsible disclosure of a bug", [("cyber-specialized", 1.0)], ["cyber-specialized/research"]),
    ("how do i pull firmware off a cheap iot device", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    # cyber-data
    ("how do i spot a brute force in a wall of ssh logs", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("whats a good feature set for detecting malicious domains", [("cyber-data", 1.0)], ["cyber-data/ml-for-security"]),
    ("my anomaly detector flags everything, too sensitive", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    ("how do i build a graph of who logged into what", [("cyber-data", 1.0)], ["cyber-data/security-graph-analytics"]),
    ("whats beaconing look like in the data", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("can i use clustering to group similar attacks", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    ("how do i label data for a phishing classifier", [("cyber-data", 1.0)], ["cyber-data/ml-for-security"]),
    ("whats a good way to correlate alerts across tools", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("how do i find the entry point from a pile of logs", [("cyber-data", 1.0)], ["cyber-data/security-graph-analytics"]),
    ("whats a reasonable threshold for failed logins before alerting", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
]

# A few messy combos, fresh.
MESSY_COMBOS: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    ("quick py script to tail a log and alert on failed logins pls", [("coding", 0.4), ("cyber-data", 0.4), ("cyber-defensive", 0.2)], ["coding/python", "cyber-data/log-analysis"]),
    ("train a lil model to flag malware from pe features", [("ai-ml", 0.45), ("cyber-data", 0.45), ("cyber-defensive", 0.1)], ["ai-ml/ml", "cyber-data/ml-for-security"]),
    ("deploy my flask api on k8s with https, where do i start", [("web-mobile", 0.35), ("cloud", 0.4), ("cyber-netsec-crypto", 0.25)], ["web-mobile/backend", "cloud/kubernetes"]),
    ("whats the big o of my recursive fib and how do i speed it in rust", [("cs-theory", 0.5), ("coding", 0.5)], ["cs-theory/complexity", "coding/rust"]),
    ("sniff the tls handshake and tell me whats weak", [("cyber-netsec-crypto", 0.55), ("networking", 0.45)], ["cyber-netsec-crypto/packet-analysis", "networking/tls"]),
    ("sql slow AND i think someones scraping the api, help with both", [("data", 0.5), ("cyber-defensive", 0.5)], ["data/databases", "cyber-defensive/detection"]),
    ("fuzz my c json parser and tell me if the crash is real", [("cyber-appsec", 0.5), ("coding", 0.3), ("cyber-offensive", 0.2)], ["cyber-appsec/fuzzing", "coding/c"]),
    ("spin up a vpc with private subnets on aws for a db", [("cloud", 0.6), ("networking", 0.4)], ["cloud/aws", "networking/tcp-ip"]),
    ("implement sha256 in go for a file hasher", [("cyber-netsec-crypto", 0.5), ("coding", 0.5)], ["cyber-netsec-crypto/cryptography", "coding/go"]),
    ("isolation forest on auth logs to catch weird logins", [("cyber-data", 0.55), ("ai-ml", 0.45)], ["cyber-data/anomaly-detection", "ai-ml/ml"]),
]

# Round-2 deepening: more messy rows for the six still-weak-on-v2.test categories
# (cyber-appsec, systems, cyber-offensive, cloud, cyber-specialized, coding).
# Fresh topics again, authored independently, deduped by the generator.
MESSY_SHORT += [
    # cyber-appsec
    ("my code review keeps missing bugs, whats a systematic way to look", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how do i set up continuous fuzzing in ci so it runs every pr", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("whats the difference between a vuln and just a weakness in code", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("found a deserialization sink, how do i tell how bad it is", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how do i write a decent fuzz harness for a protocol parser", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("sast says tainted input reaches exec(), is that always rce", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("how do i audit crypto usage across a whole codebase", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("my dast scan found nothing, is the app clean or the scan weak", [("cyber-appsec", 1.0)], ["cyber-appsec/dast"]),
    ("how do i find logic bugs that the scanners never catch", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("whats a good way to triage which sast findings to fix first", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    # systems
    ("my program pins one core at 100% how do i spread the work", [("systems", 1.0)], ["systems/concurrency"]),
    ("whats the page cache, my free mem looks scary low but isnt", [("systems", 1.0)], ["systems/linux"]),
    ("how do i profile where my c program actually spends time", [("systems", 1.0)], ["systems/linux"]),
    ("mmap vs plain read for a huge file, which is faster and why", [("systems", 1.0)], ["systems/linux"]),
    ("my daemon leaves zombie children, how do i reap them", [("systems", 1.0)], ["systems/linux"]),
    ("whats the actual difference between fork and exec", [("systems", 1.0)], ["systems/linux"]),
    ("how do i pin a thread to a specific cpu core", [("systems", 1.0)], ["systems/concurrency"]),
    ("my embedded board resets at random, is it the watchdog", [("systems", 1.0)], ["systems/embedded"]),
    ("whats a memory barrier and when do i actually need one", [("systems", 1.0)], ["systems/concurrency"]),
    ("how do i even start debugging a kernel oops", [("systems", 1.0)], ["systems/kernel"]),
    # cyber-offensive
    ("how do i pivot through a compromised host into the internal net", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("whats a solid wordlist for brute forcing web directories", [("cyber-offensive", 1.0)], ["cyber-offensive/recon"]),
    ("how do i get past a waf thats blocking my sqli", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("theres a file upload, how do i turn it into code execution", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("whats ret2libc and when do i reach for it", [("cyber-offensive", 1.0)], ["cyber-offensive/binary-exploitation"]),
    ("found a writable /etc/passwd, how do i escalate with it", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("whats token impersonation on windows and how do i abuse it", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("how do i enumerate domain users once i can reach ldap", [("cyber-offensive", 1.0)], ["cyber-offensive/enumeration"]),
    ("whats a golden ticket and why is it so bad", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("how do i keep access after a reboot without being obvious", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    # cloud
    ("my ecs task cant pull the image, is it a perms thing", [("cloud", 1.0)], ["cloud/aws"]),
    ("how do i let a pod read an s3 bucket without hardcoding keys", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("nodeport vs loadbalancer service, which do i want", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("terraform recreates my resource every apply, how do i stop it", [("cloud", 1.0)], ["cloud/terraform-iac"]),
    ("how do i cache docker layers in github actions properly", [("cloud", 1.0)], ["cloud/ci-cd"]),
    ("whats a sane autoscaling policy for bursty traffic", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("my helm upgrade died halfway, how do i recover the release", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("how do i rotate db secrets with no downtime", [("cloud", 1.0)], ["cloud/sre"]),
    ("fargate vs ec2 for ecs, whats the tradeoff", [("cloud", 1.0)], ["cloud/aws"]),
    ("theres a mystery charge on my cloud bill, how do i find it", [("cloud", 1.0)], ["cloud/observability"]),
    # cyber-specialized
    ("how would i test an iot doorbell for vulns", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats a good approach for auditing a cloud config", [("cyber-specialized", 1.0)], ["cyber-specialized/cloud-sec"]),
    ("how do i map a companys exposed assets with just osint", [("cyber-specialized", 1.0)], ["cyber-specialized/osint"]),
    ("how do i get past the biometric lock on a test phone", [("cyber-specialized", 1.0)], ["cyber-specialized/mobile-sec"]),
    ("whats actually involved in a physical pentest", [("cyber-specialized", 1.0)], ["cyber-specialized/methodology"]),
    ("how do i sniff zigbee traffic from a smart home", [("cyber-specialized", 1.0)], ["cyber-specialized/wireless-sec"]),
    ("how do i build a phishing landing page for an authorized test", [("cyber-specialized", 1.0)], ["cyber-specialized/social-engineering"]),
    ("how do i safely assess a scada hmi without breaking it", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats the real risk with default creds on ip cameras", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("how do i pull and analyze the firmware off a home router", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    # coding
    ("my recursive fn blows the stack, how do i make it iterative", [("coding", 1.0)], ["coding/python"]),
    ("whats the idiomatic way to handle errors in rust, Result everywhere?", [("coding", 1.0)], ["coding/rust"]),
    ("how do i parse json in c without pulling a library", [("coding", 1.0)], ["coding/c"]),
    ("does the python gil mean threads are pointless for cpu work", [("coding", 1.0)], ["coding/python"]),
    ("how do i write a generic function in go now that we have them", [("coding", 1.0)], ["coding/go"]),
    ("trait object vs generic in rust, when which", [("coding", 1.0)], ["coding/rust"]),
    ("my regex does catastrophic backtracking, how do i rewrite it", [("coding", 1.0)], ["coding/python"]),
    ("whats a clean way to do dependency injection in java", [("coding", 1.0)], ["coding/java"]),
    ("shallow vs deep copy, i keep getting bitten by this", [("coding", 1.0)], ["coding/python"]),
    ("how do i memoize a function in python, functools?", [("coding", 1.0)], ["coding/python"]),
]

# Near-miss combos for the confusable pairs d9 flagged: appsec vs offensive,
# and systems vs coding vs cloud. Each spans the pair deliberately so the router
# learns the boundary instead of collapsing one into the other.
MESSY_COMBOS += [
    # appsec <-> offensive (the audit/exploit boundary)
    ("fuzz this image parser and write a poc for whatever crash it finds", [("cyber-appsec", 0.55), ("cyber-offensive", 0.45)], ["cyber-appsec/fuzzing", "cyber-offensive/binary-exploitation"]),
    ("audit the auth code then actually try to bypass it to prove it", [("cyber-appsec", 0.5), ("cyber-offensive", 0.5)], ["cyber-appsec/code-audit", "cyber-offensive/web-exploitation"]),
    ("sast flagged an sqli, confirm it by exploiting the endpoint", [("cyber-appsec", 0.45), ("cyber-offensive", 0.55)], ["cyber-appsec/sast", "cyber-offensive/web-exploitation"]),
    ("review this binary for bugs and check which are exploitable", [("cyber-appsec", 0.55), ("cyber-offensive", 0.45)], ["cyber-appsec/bug-finding", "cyber-offensive/binary-exploitation"]),
    ("found a use-after-free while fuzzing, turn it into code exec", [("cyber-appsec", 0.45), ("cyber-offensive", 0.55)], ["cyber-appsec/fuzzing", "cyber-offensive/exploit-dev"]),
    ("static-analyze the parser then build a crashing input for the worst finding", [("cyber-appsec", 0.5), ("cyber-offensive", 0.5)], ["cyber-appsec/sast", "cyber-offensive/zeroday"]),
    # systems <-> coding (write low-level systems code)
    ("write a lock-free ring buffer in rust for my audio thread", [("systems", 0.5), ("coding", 0.5)], ["systems/concurrency", "coding/rust"]),
    ("implement an epoll-based event loop in c", [("systems", 0.5), ("coding", 0.5)], ["systems/linux", "coding/c"]),
    ("code a work-stealing thread pool in cpp", [("systems", 0.5), ("coding", 0.5)], ["systems/concurrency", "coding/cpp"]),
    ("write a custom bump allocator in c and explain the alignment", [("systems", 0.5), ("coding", 0.5)], ["systems/linux", "coding/c"]),
    # systems <-> cloud (kernel/host tuning for cloud infra)
    ("tune linux tcp buffers and ulimits on my k8s worker nodes", [("systems", 0.5), ("cloud", 0.5)], ["systems/linux", "cloud/kubernetes"]),
    ("my container cant use hugepages, is it a kernel or k8s config", [("systems", 0.5), ("cloud", 0.5)], ["systems/kernel", "cloud/kubernetes"]),
    # coding <-> cloud (write code that drives cloud)
    ("write a go microservice and containerize it for k8s", [("coding", 0.5), ("cloud", 0.5)], ["coding/go", "cloud/kubernetes"]),
    ("python script to auto-scale my ec2 fleet on a metric", [("coding", 0.5), ("cloud", 0.5)], ["coding/python", "cloud/aws"]),
    # systems <-> coding <-> cloud (triple near-miss)
    ("profile my rust service, fix the hot path, then ship it to k8s", [("systems", 0.34), ("coding", 0.33), ("cloud", 0.33)], ["systems/linux", "coding/rust", "cloud/kubernetes"]),
]

# Round-3: anchor the offensive category away from appsec with prompts that are
# UNAMBIGUOUSLY offensive action (exploit / post-exploit / C2 / evasion), using
# no audit/review verbs that lean appsec. Plus fresh coding + web-mobile. These
# are red-team routing labels (one-line questions a pentester asks), not exploit
# code. Fresh topics, deduped, authored independently.
MESSY_SHORT += [
    # cyber-offensive — exploitation (act, don't audit)
    ("i have this sqli, how do i actually pop a shell with it not just detect", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("got an rce on a box, how do i upgrade it to an interactive tty", [("cyber-offensive", 1.0)], ["cyber-offensive/exploit-dev"]),
    ("how do i weaponize this deserialization bug into code exec", [("cyber-offensive", 1.0)], ["cyber-offensive/exploit-dev"]),
    ("xss on an admin panel, how do i steal the session and take it over", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("how do i chain this ssrf into hitting the cloud metadata endpoint", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    # cyber-offensive — post-exploitation
    ("low-priv shell on windows, how do i dump lsass for creds", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("how do i harvest cached credentials after landing on a host", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("whats the fastest path to domain admin from a single foothold", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("how do i pivot into the next subnet through this compromised host", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("how do i exfil a database quietly so it doesnt look like a spike", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    # cyber-offensive — C2 and evasion
    ("how do i set up a c2 beacon with sane sleep and jitter", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("whats a good c2 channel that blends in with normal https traffic", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("how do i stage my payload so the first-stage dropper stays tiny", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("how do i set up persistence that survives a reboot on the target", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("how do i get my implant past defender on a red team engagement", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("whats a reasonable way to evade edr during post-ex", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("how do i bypass amsi so my script actually runs", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    # coding — fresh
    ("whats the borrow checker on about with 'cannot borrow as mutable'", [("coding", 1.0)], ["coding/rust"]),
    ("how do i reverse a string in place in c", [("coding", 1.0)], ["coding/c"]),
    ("python whats the clean way to flatten a nested list", [("coding", 1.0)], ["coding/python"]),
    ("tuple vs list in python, is one actually faster", [("coding", 1.0)], ["coding/python"]),
    ("kotlin how do i handle a nullable without it crashing", [("coding", 1.0)], ["coding/kotlin"]),
    ("go why does my map access panic sometimes", [("coding", 1.0)], ["coding/go"]),
    ("how do i write a python decorator that takes arguments", [("coding", 1.0)], ["coding/python"]),
    ("pointer vs reference in cpp, when do i use which", [("coding", 1.0)], ["coding/cpp"]),
    ("how do i pattern match on an enum in rust", [("coding", 1.0)], ["coding/rust"]),
    ("my js promise never resolves, did i forget to return it", [("coding", 1.0)], ["coding/js-ts"]),
    # web-mobile — fresh
    ("my fetch gets an opaque response, is this cors again", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do i lazy load images so the page paints sooner", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("react state wont update when i mutate the array in place", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do i keep a user logged in across page refreshes", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("my recyclerview scrolls janky on android, whats the fix", [("web-mobile", 1.0)], ["web-mobile/android"]),
    ("how do i handle deep links into specific screens in an ios app", [("web-mobile", 1.0)], ["web-mobile/ios"]),
    ("whats the right status code to return for a failed validation", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("how do i debounce autosave in my web form so it doesnt spam", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("my service worker keeps serving stale content, how do i bust it", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do i upload a big file from the browser without timing out", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
]

# A few messy OOS, fresh.
MESSY_OOS: list[str] = [
    "whats a good marinade for chicken thighs",
    "recommend a cosy show to binge this weekend",
    "how do i get my sourdough starter going again",
    "whats a cheap but nice anniversary idea",
    "how do i stop my glasses fogging with a mask",
    "any tips for a long haul flight with a toddler",
    "whats the best way to reheat pizza",
    "how do i keep ants out of the kitchen",
    "recommend a relaxing weekend getaway near the coast",
    "whats a good icebreaker for a work offsite",
]
