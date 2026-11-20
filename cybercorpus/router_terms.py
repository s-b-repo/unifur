#!/usr/bin/env python3
"""Domain VOCABULARY per category/leaf for the router training generator.

The n-gram meta-router's in-scope top-1 stalled at 0.48 because it never saw the
domain terms a real user uses — it was trained on leaf *labels* ("Rust", "DNS"),
not the vocabulary of each field. This is a hand-authored term bank: dozens of
concepts, methods and tools per category, grouped by the canonical leaf id, so a
prompt like "What is osmosis?" trains science/biology with real biology vocab.

IMPORTANT: these lists were written from general domain knowledge (textbook
glossaries, course syllabi, common tooling) WITHOUT reading router-eval.jsonl —
copying the eval's words would leak the held-out set. The generator still dedupes
generated prompts against the eval by hash, which never exposes its wording here.

TERM_BANK[category_id][leaf_id] = [terms...]. Weak categories (science, ai-ml,
cyber-defensive, data) are the richest, per the router owner's priority.
"""

from __future__ import annotations

# Per-category prompt target; weak categories get more (router owner asked 80-100).
CATEGORY_TARGET = {
    "science": 90, "ai-ml": 90, "cyber-defensive": 90, "data": 90,
}
DEFAULT_TARGET = 55

TERM_BANK: dict[str, dict[str, list[str]]] = {
    "science": {
        "biology": ["photosynthesis", "mitosis", "meiosis", "osmosis", "enzymes", "DNA replication",
                    "natural selection", "the cell membrane", "homeostasis", "cellular respiration",
                    "the nitrogen cycle", "protein synthesis", "genetic mutation", "the food chain",
                    "ecosystems", "mitochondria", "chromosomes", "evolution by selection", "diffusion"],
        "chemistry": ["covalent bonds", "oxidation and reduction", "the pH scale", "molarity", "catalysts",
                      "isotopes", "the periodic table", "acids and bases", "electron shells", "stoichiometry",
                      "ionic bonding", "chemical equilibrium", "exothermic reactions", "the mole concept",
                      "electronegativity", "titration", "valence electrons"],
        "physics": ["Newton's laws of motion", "momentum", "entropy", "electromagnetism", "relativity",
                    "the laws of thermodynamics", "wave-particle duality", "friction", "gravitational force",
                    "kinetic energy", "the Doppler effect", "refraction", "nuclear fission", "capacitance",
                    "angular velocity", "the photoelectric effect"],
        "earth-space": ["plate tectonics", "the water cycle", "black holes", "the greenhouse effect",
                        "erosion", "orbital mechanics", "stellar fusion", "the rock cycle", "tides",
                        "the ozone layer", "seismic waves", "the solar wind", "glaciation"],
        "general-science": ["the scientific method", "a controlled experiment", "a hypothesis", "peer review"],
    },
    "ai-ml": {
        "ml": ["gradient descent", "overfitting", "regularization", "cross-validation", "random forests",
               "gradient boosting", "k-means clustering", "principal component analysis", "feature engineering",
               "the bias-variance tradeoff", "logistic regression", "support vector machines", "a confusion matrix",
               "precision and recall", "the ROC curve", "ensemble learning", "hyperparameter tuning", "label leakage"],
        "deep-learning": ["backpropagation", "convolutional neural networks", "the attention mechanism",
                          "dropout", "batch normalization", "residual connections", "an autoencoder",
                          "a recurrent neural network", "LSTMs", "the vanishing gradient problem",
                          "weight initialization", "a loss function", "an activation function", "fine-tuning"],
        "nlp": ["word embeddings", "tokenization", "the transformer architecture", "BERT", "beam search",
                "perplexity", "named entity recognition", "a language model", "self-attention", "positional encoding"],
        "cv": ["image segmentation", "object detection", "a convolution kernel", "data augmentation",
               "a feature map", "transfer learning for vision", "image classification"],
        "rl": ["Q-learning", "a reward function", "the exploration-exploitation tradeoff", "policy gradients",
               "a Markov decision process", "temporal-difference learning"],
        "mlops": ["model serving", "a feature store", "model drift", "an experiment tracker",
                  "a training pipeline", "model quantization"],
    },
    "cyber-defensive": {
        "siem": ["a SIEM", "log correlation", "Splunk", "the ELK stack", "Wazuh", "alert tuning",
                 "a correlation rule", "log aggregation", "normalisation of events"],
        "detection": ["YARA rules", "Sigma rules", "the MITRE ATT&CK framework", "indicators of compromise",
                      "Sysmon", "Windows event logs", "an intrusion detection system", "anomaly-based detection",
                      "signature-based detection", "detection engineering", "a detection-as-code pipeline"],
        "incident-response": ["incident response", "the containment phase", "an IR playbook", "triage",
                              "the kill chain", "eradication and recovery", "a tabletop exercise"],
        "forensics": ["memory forensics", "disk imaging", "timeline analysis", "Volatility", "Velociraptor",
                      "the chain of custody", "file carving", "a forensic image", "artifact analysis"],
        "malware-analysis": ["malware sandboxing", "static analysis of a binary", "dynamic analysis",
                             "unpacking a packed sample", "a YARA signature for malware", "C2 beaconing"],
        "threat-intel": ["threat intelligence", "a threat actor profile", "TTPs", "an IOC feed", "attribution"],
        "blue-team": ["the blue team", "a purple team exercise", "network segmentation", "defence in depth"],
        "hardening": ["system hardening", "the principle of least privilege", "a CIS benchmark", "attack surface reduction"],
        "reverse-engineering": ["reverse engineering a binary", "disassembly", "Ghidra", "control-flow analysis"],
    },
    "data": {
        "databases": ["a SQL join", "an index", "query optimisation", "database normalisation", "ACID transactions",
                      "a foreign key", "denormalisation", "a B-tree index", "write-ahead logging", "a materialised view",
                      "a window function", "a primary key", "referential integrity", "a stored procedure", "query planning"],
        "data-engineering": ["an ETL pipeline", "a data warehouse", "a star schema", "change data capture",
                            "a data lake", "Apache Airflow", "dbt", "the Parquet format", "data partitioning",
                            "a slowly changing dimension", "batch ingestion", "schema evolution"],
        "distributed-systems": ["sharding", "replication", "the CAP theorem", "eventual consistency",
                               "a distributed transaction", "consensus with Raft", "leader election", "quorum reads"],
        "streaming": ["Apache Kafka", "stream processing", "a windowed aggregation", "exactly-once semantics",
                      "backpressure", "a message queue", "event-time processing"],
        "analytics": ["OLAP", "columnar storage", "a cube", "a rollup", "an aggregation pipeline", "cardinality estimation"],
    },
    # Moderate banks for the stronger categories (broaden vocab without over-investing).
    "coding": {
        "rust": ["the borrow checker", "ownership and lifetimes", "a trait object", "pattern matching", "the Result type"],
        "python": ["a list comprehension", "a generator", "a decorator", "the GIL", "duck typing"],
        "c": ["a dangling pointer", "manual memory management", "a buffer overflow", "pointer arithmetic", "undefined behaviour"],
        "go": ["a goroutine", "a channel", "the defer statement", "a slice", "error wrapping"],
        "js-ts": ["a closure", "the event loop", "a promise", "async/await", "prototypal inheritance"],
    },
    "systems": {
        "linux": ["a system call", "virtual memory", "a page fault", "the scheduler", "a cgroup"],
        "kernel": ["a kernel module", "interrupt handling", "a device driver", "kernel space vs user space"],
        "concurrency": ["a race condition", "a mutex", "a deadlock", "atomic operations", "a memory barrier"],
        "embedded": ["an interrupt service routine", "a watchdog timer", "DMA", "firmware flashing"],
    },
    "networking": {
        "tcp-ip": ["the three-way handshake", "congestion control", "the TCP window", "a retransmission"],
        "dns": ["a DNS resolver", "a zone transfer", "DNS caching", "an A record", "DNS over HTTPS"],
        "http": ["an HTTP header", "a status code", "HTTP keep-alive", "chunked transfer encoding"],
        "tls": ["the TLS handshake", "a certificate chain", "forward secrecy", "mutual TLS"],
        "routing-bgp": ["BGP", "a routing table", "an autonomous system", "route flapping"],
    },
    "cloud": {
        "aws": ["an S3 bucket", "an IAM policy", "a VPC", "an EC2 instance", "a security group"],
        "kubernetes": ["a pod", "an ingress", "a deployment", "a service mesh", "a readiness probe"],
        "docker": ["a container image", "a layer cache", "a multi-stage build", "a volume mount"],
        "terraform-iac": ["infrastructure as code", "a terraform plan", "state drift", "a module"],
        "ci-cd": ["a CI pipeline", "a deployment rollback", "a canary release", "a build artifact"],
    },
    "web-mobile": {
        "frontend": ["the virtual DOM", "CSS flexbox", "a React hook", "client-side routing", "hydration"],
        "backend": ["a REST endpoint", "middleware", "an ORM", "connection pooling", "rate limiting"],
        "api-design": ["pagination", "idempotency", "API versioning", "a webhook", "an OpenAPI spec"],
        "android": ["an activity lifecycle", "an intent", "a content provider", "a background service"],
    },
    "cs-theory": {
        "algorithms": ["dynamic programming", "a greedy algorithm", "binary search", "Dijkstra's algorithm", "backtracking"],
        "data-structures": ["a hash table", "a balanced binary tree", "a heap", "a trie", "a disjoint-set"],
        "complexity": ["big-O notation", "NP-completeness", "amortised analysis", "the halting problem"],
        "compilers": ["lexical analysis", "a parse tree", "register allocation", "a context-free grammar"],
    },
    "math": {
        "algebra": ["a quadratic equation", "factoring polynomials", "a system of equations", "the binomial theorem"],
        "calculus": ["a derivative", "an integral", "the chain rule", "a limit", "a Taylor series"],
        "linear-algebra": ["a matrix inverse", "eigenvalues", "a dot product", "matrix multiplication", "a vector space"],
        "probability": ["conditional probability", "Bayes' theorem", "a normal distribution", "expected value"],
        "number-theory": ["modular arithmetic", "prime factorisation", "the greatest common divisor", "Euler's totient"],
    },
    "cyber-offensive": {
        "recon": ["subdomain enumeration", "OSINT gathering", "port scanning", "service fingerprinting", "banner grabbing"],
        "web-exploitation": ["SQL injection", "cross-site scripting", "server-side request forgery", "an IDOR", "a CSRF attack"],
        "binary-exploitation": ["a buffer overflow", "return-oriented programming", "a use-after-free", "a format-string bug"],
        "privesc": ["privilege escalation", "a SUID binary", "token impersonation", "a sudo misconfiguration"],
        "c2": ["a command-and-control channel", "beaconing", "a reverse shell", "DNS tunnelling"],
    },
    "cyber-appsec": {
        "fuzzing": ["fuzzing", "a fuzzer harness", "coverage-guided fuzzing", "a crash triage", "sanitizers"],
        "sast": ["static application security testing", "taint analysis", "a false positive", "a data-flow rule"],
        "code-audit": ["a security code review", "input validation", "an injection sink", "secure defaults"],
        "bug-finding": ["a vulnerability", "a proof of concept", "root-cause analysis", "a security regression"],
    },
    "cyber-netsec-crypto": {
        "cryptography": ["AES encryption", "RSA", "a digital signature", "a hash function", "a nonce", "key exchange"],
        "cryptanalysis": ["a padding oracle", "a known-plaintext attack", "frequency analysis", "a weak cipher"],
        "pki": ["a certificate authority", "a certificate signing request", "certificate revocation", "the chain of trust"],
        "firewalls": ["a firewall rule", "stateful inspection", "an access control list", "egress filtering"],
        "packet-analysis": ["a packet capture", "Wireshark", "a protocol dissector", "a pcap file"],
    },
    "cyber-specialized": {
        "osint": ["open-source intelligence", "a Google dork", "metadata analysis", "a breach database lookup"],
        "social-engineering": ["a phishing campaign", "pretexting", "a spear-phishing email", "a cloned login page"],
        "mobile-sec": ["an APK analysis", "certificate pinning bypass", "an insecure data store", "a jailbreak detection bypass"],
        "cloud-sec": ["a misconfigured S3 bucket", "an over-permissive IAM role", "cloud privilege escalation"],
        "wireless-sec": ["a WPA2 handshake capture", "an evil twin access point", "a deauthentication attack"],
    },
    "cyber-data": {
        "log-analysis": ["log correlation", "parsing firewall logs", "a brute-force pattern in logs", "beaconing in netflow"],
        "anomaly-detection": ["an isolation forest", "a baseline of normal behaviour", "an outlier score", "unsupervised anomaly detection"],
        "ml-for-security": ["a malware classifier", "feature extraction from binaries", "a phishing-URL model", "a fraud-detection model"],
        "security-graph-analytics": ["an attack graph", "graph-based threat detection", "entity resolution", "a lateral-movement graph"],
    },
}

# Deepened vocabulary for the categories still weak on router-eval-v2
# (systems, cloud, web-mobile, networking, cyber-specialized, math, cyber-offensive).
_SUPPLEMENT: dict[str, dict[str, list[str]]] = {
    "systems": {
        "linux": ["a file descriptor", "an inode", "epoll", "cgroups", "a namespace", "the OOM killer",
                  "swap space", "strace", "systemd", "the ELF format", "a signal handler", "mmap"],
        "kernel": ["a spinlock", "the CFS scheduler", "a system call", "a loadable kernel module", "procfs", "eBPF"],
        "concurrency": ["a lock-free queue", "compare-and-swap", "false sharing", "memory ordering", "the actor model"],
        "windows-internals": ["the registry", "DLL injection", "the PE format", "a handle", "structured exception handling"],
        "drivers": ["a character device", "an ioctl", "a DMA transfer", "a kernel driver"],
        "embedded": ["GPIO", "SPI", "I2C", "a UART", "a hardware interrupt"],
        "firmware": ["a bootloader", "an OTA update", "a flash image"],
        "rtos": ["task scheduling", "priority inversion", "a real-time deadline"],
    },
    "cloud": {
        "aws": ["AWS Lambda", "DynamoDB", "CloudFormation", "Route 53", "ECS", "an RDS instance", "CloudWatch", "SQS"],
        "kubernetes": ["a StatefulSet", "a ConfigMap", "RBAC", "a Helm chart", "node affinity", "horizontal autoscaling", "a sidecar"],
        "docker": ["a Dockerfile", "a base image", "docker-compose", "a bind mount", "a container registry"],
        "terraform-iac": ["a provider", "a remote backend", "a workspace", "a module", "a resource graph"],
        "ci-cd": ["a build runner", "a matrix build", "artifact caching", "a deployment gate"],
        "sre": ["an SLO", "an error budget", "a postmortem", "toil", "on-call rotation"],
        "observability": ["distributed tracing", "a metrics dashboard", "structured logging", "cardinality"],
        "gcp": ["BigQuery", "Cloud Run", "GKE", "a service account"],
        "azure": ["a resource group", "Azure Functions", "a blob container"],
    },
    "web-mobile": {
        "frontend": ["the DOM", "a web component", "lazy loading", "a service worker", "tree shaking", "a bundler", "hydration"],
        "backend": ["a session", "a JWT", "middleware", "an API gateway", "CORS", "GraphQL", "connection pooling"],
        "api-design": ["REST", "an idempotency key", "rate limiting", "API versioning", "a webhook", "pagination"],
        "android": ["an intent", "a fragment", "Jetpack Compose", "a broadcast receiver", "the activity lifecycle"],
        "ios": ["SwiftUI", "a view controller", "Core Data", "an optional"],
        "browsers": ["the render pipeline", "the event loop", "a cookie", "local storage"],
    },
    "networking": {
        "tcp-ip": ["the OSI model", "a socket", "congestion control", "the MTU", "ARP", "ICMP", "a three-way handshake"],
        "dns": ["a TXT record", "DNSSEC", "a recursive resolver", "an MX record", "DNS caching"],
        "http": ["a reverse proxy", "caching headers", "HTTP/3", "a status code", "keep-alive"],
        "tls": ["a cipher suite", "SNI", "OCSP stapling", "a certificate chain"],
        "routing-bgp": ["OSPF", "a default route", "CIDR notation", "an autonomous system"],
        "wireless": ["WPA3", "a wifi channel", "an SSID", "signal interference"],
        "sdn": ["a VLAN", "an overlay network", "VXLAN", "a software-defined network"],
        "protocols": ["NAT", "a load balancer", "DHCP", "a VPN tunnel"],
    },
    "cyber-specialized": {
        "osint": ["Shodan", "Google dorking", "a WHOIS lookup", "social-media reconnaissance", "a breach dump"],
        "social-engineering": ["a phishing kit", "vishing", "baiting", "tailgating", "a pretext call"],
        "mobile-sec": ["Frida", "Objection", "a jailbreak", "root detection bypass", "certificate pinning"],
        "cloud-sec": ["IAM privilege escalation", "a misconfigured bucket", "SSRF to the metadata service", "an exposed key"],
        "wireless-sec": ["a WPA handshake capture", "aircrack", "an evil-twin AP", "a rogue access point", "a deauth attack"],
        "iot-ics-scada": ["Modbus", "a PLC", "firmware extraction", "an industrial protocol"],
        "methodology": ["a pentest methodology", "scoping an engagement", "writing the report", "rules of engagement"],
        "research": ["vulnerability research", "a CVE", "responsible disclosure", "a bug bounty"],
    },
    "math": {
        "linear-algebra": ["a matrix inverse", "eigenvectors", "singular value decomposition", "a basis", "the rank of a matrix", "a linear transformation"],
        "calculus": ["a limit", "partial derivatives", "the fundamental theorem of calculus", "a gradient", "an improper integral"],
        "probability": ["a probability distribution", "variance", "a random variable", "the law of large numbers", "a Markov chain"],
        "statistics": ["a p-value", "a confidence interval", "linear regression", "hypothesis testing", "the central limit theorem"],
        "optimization": ["gradient descent", "a convex function", "Lagrange multipliers", "a local minimum", "linear programming"],
        "number-theory": ["prime numbers", "the greatest common divisor", "modular arithmetic", "Fermat's little theorem"],
        "geometry": ["a vector", "the dot product", "a rotation matrix", "the Pythagorean theorem"],
        "logic": ["a proof by contradiction", "mathematical induction", "a truth table", "a quantifier"],
        "graph-theory": ["a spanning tree", "the shortest path", "a bipartite graph", "graph coloring"],
        "algebra": ["a system of equations", "the quadratic formula", "a logarithm", "a function's domain"],
    },
    "cyber-offensive": {
        "recon": ["subdomain enumeration", "nmap", "directory brute-forcing", "passive reconnaissance", "a DNS zone transfer"],
        "scanning": ["a port scan", "a vulnerability scan", "service version detection", "a SYN scan"],
        "enumeration": ["SMB enumeration", "LDAP enumeration", "user enumeration", "share enumeration"],
        "exploit-dev": ["a ROP chain", "shellcode", "a proof-of-concept exploit", "a NOP sled"],
        "zeroday": ["a zero-day", "fuzzing for bugs", "a memory corruption", "an n-day exploit"],
        "web-exploitation": ["cross-site scripting", "SQL injection", "SSRF", "XXE", "an IDOR", "insecure deserialization"],
        "binary-exploitation": ["a stack overflow", "a heap overflow", "a format-string bug", "an ASLR bypass", "a GOT overwrite"],
        "privesc": ["a SUID binary", "sudo misconfiguration", "a kernel exploit", "a writable cron job"],
        "lateral-movement": ["pass-the-hash", "pivoting", "RDP hijacking", "a scheduled-task move"],
        "c2": ["a beacon", "a reverse shell", "DNS tunnelling", "a dropper", "a staged payload"],
    },
}
for _c, _leaves in _SUPPLEMENT.items():
    for _leaf, _terms in _leaves.items():
        TERM_BANK.setdefault(_c, {}).setdefault(_leaf, []).extend(_terms)
# These seven get the weak-category target too.
CATEGORY_TARGET.update({_c: 90 for _c in _SUPPLEMENT})
