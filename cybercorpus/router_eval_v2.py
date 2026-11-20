#!/usr/bin/env python3
"""router-eval-v2: a larger, messier held-out eval for the 3-layer router.

The v1 eval (n=56) is too small — one prompt moves top-1 by 1.8 points. This is
a bigger held-out set in a DELIBERATELY DIFFERENT STYLE from the training
generator: messy casing, typos, run-ons and terse fragments, questions mixed
with imperatives and "help me" phrasing. >=15 per category, ~15% out-of-scope,
>=20 cross-category combos.

Authored by hand from general knowledge, NOT from the term bank and NOT by
reading the training file — then hash-deduped against both the train set and v1
eval, so it is a genuinely independent measurement. router-eval.jsonl is left
as-is for continuity. Schema: {"id","text","categories":[{"id","weight"}...],
"subfields":[...],"style":"short"|"combo"|"oos"}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# (text, [(cat, weight)...], [subfield...]) — messy, varied, user-real.
P: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    # coding
    ("my rust prog wont compile, borrow checker yelling about a moved value???", [("coding", 1.0)], ["coding/rust"]),
    ("help i need to reverse a linked list in C but keep getting null ptr", [("coding", 1.0)], ["coding/c"]),
    ("whats the deal with async await in python, is it actually parallel", [("coding", 1.0)], ["coding/python"]),
    ("Go: how do i wait for a bunch of goroutines to finish before returning", [("coding", 1.0)], ["coding/go"]),
    ("js closures are melting my brain can someone explain like im 5", [("coding", 1.0)], ["coding/js-ts"]),
    ("Write me a quick bash one-liner to find the 10 biggest files", [("coding", 1.0)], ["coding/shell"]),
    ("why is my java program throwing ConcurrentModificationException", [("coding", 1.0)], ["coding/java"]),
    ("best way to format a date string in php?? the docs are a mess", [("coding", 1.0)], ["coding/php"]),
    ("how do interfaces work in typescript vs regular js", [("coding", 1.0)], ["coding/js-ts"]),
    ("i want to learn systems programming, is rust or c++ better to start", [("coding", 1.0)], ["coding/cpp"]),
    ("segfault in my C code only when optimizations are on, wtf", [("coding", 1.0)], ["coding/c"]),
    ("explain generics in go, do they even have them now", [("coding", 1.0)], ["coding/go"]),
    # systems
    ("my linux box is thrashing, how do i find whats eating all the RAM", [("systems", 1.0)], ["systems/linux"]),
    ("what actually happens during a context switch", [("systems", 1.0)], ["systems/kernel"]),
    ("writing my first device driver, where do i even start", [("systems", 1.0)], ["systems/drivers"]),
    ("deadlock between two threads each holding a lock the other wants, how to fix", [("systems", 1.0)], ["systems/concurrency"]),
    ("whats a page fault and why are mine so slow", [("systems", 1.0)], ["systems/linux"]),
    ("embedded: how do i debug an ISR that never fires", [("systems", 1.0)], ["systems/embedded"]),
    ("difference between a process and a thread, for real this time", [("systems", 1.0)], ["systems/concurrency"]),
    ("how to flash firmware onto an stm32 without a debugger", [("systems", 1.0)], ["systems/firmware"]),
    # networking
    ("my packets are getting dropped somewhere, how do i trace the path", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("dns resolves fine on my laptop but not the server, halp", [("networking", 1.0)], ["networking/dns"]),
    ("whats the diff between http 1.1 and 2, do i care", [("networking", 1.0)], ["networking/http"]),
    ("setting up a site, do i really need tls everywhere", [("networking", 1.0)], ["networking/tls"]),
    ("explain bgp like im not a network engineer", [("networking", 1.0)], ["networking/routing-bgp"]),
    ("why does my wifi drop every 10 min, is it interference", [("networking", 1.0)], ["networking/wireless"]),
    ("how does a load balancer actually distribute traffic", [("networking", 1.0)], ["networking/protocols"]),
    ("what port is this service probably on, its a mail thing", [("networking", 1.0)], ["networking/protocols"]),
    # cloud
    ("my aws bill tripled last month and i have no idea why", [("cloud", 1.0)], ["cloud/aws"]),
    ("k8s pod keeps restarting, CrashLoopBackOff, send help", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("whats the point of docker if i already have a VM", [("cloud", 1.0)], ["cloud/docker"]),
    ("terraform apply wants to destroy my database?!? how do i stop that", [("cloud", 1.0)], ["cloud/terraform-iac"]),
    ("how do i do zero downtime deploys", [("cloud", 1.0)], ["cloud/ci-cd"]),
    ("gcp vs azure for a small startup, which is cheaper", [("cloud", 1.0)], ["cloud/gcp"]),
    ("my CI pipeline takes 40 minutes, how do i speed it up", [("cloud", 1.0)], ["cloud/ci-cd"]),
    ("what does SRE even mean, is it just ops with a fancy name", [("cloud", 1.0)], ["cloud/sre"]),
    # data
    ("my sql query is super slow, added an index but no change??", [("data", 1.0)], ["data/databases"]),
    ("whats normalization and do i actually need 3rd normal form", [("data", 1.0)], ["data/databases"]),
    ("explain the CAP theorem, im designing a distributed thing", [("data", 1.0)], ["data/distributed-systems"]),
    ("kafka consumer lag keeps growing, how do i catch up", [("data", 1.0)], ["data/streaming"]),
    ("building an ETL, should i use airflow or just cron", [("data", 1.0)], ["data/data-engineering"]),
    ("what is a star schema and when do i use it", [("data", 1.0)], ["data/analytics"]),
    ("sharding vs replication, i always mix them up", [("data", 1.0)], ["data/distributed-systems"]),
    ("how do window functions work in sql, the OVER thing", [("data", 1.0)], ["data/databases"]),
    # web-mobile
    ("why wont my div center, ive tried everything, flexbox hates me", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("react re-renders too much, how do i stop it", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do i add pagination to a rest api without it being slow", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("my android app crashes on rotate, something about the activity lifecycle", [("web-mobile", 1.0)], ["web-mobile/android"]),
    ("whats the right way to handle auth in a single page app", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("ios: app rejected for using a private api, now what", [("web-mobile", 1.0)], ["web-mobile/ios"]),
    ("should i use an ORM or raw sql for my backend", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("how do webhooks work, do i need to poll", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("css grid vs flexbox when do i use which", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    # ai-ml
    ("my model gets 99% train acc but 60% on test, overfitting right", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("explain attention in transformers, the Q K V thing confuses me", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("whats a good baseline model for tabular data", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("how do embeddings capture meaning, like actually", [("ai-ml", 1.0)], ["ai-ml/nlp"]),
    ("my loss is nan after a few steps, what did i break", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("difference between precision and recall, i always forget", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("how do i detect model drift in production", [("ai-ml", 1.0)], ["ai-ml/mlops"]),
    ("object detection vs image classification whats the difference", [("ai-ml", 1.0)], ["ai-ml/cv"]),
    ("whats reinforcement learning good for besides games", [("ai-ml", 1.0)], ["ai-ml/rl"]),
    # cs-theory
    ("is my algorithm O(n log n) or O(n^2), i cant tell", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("when should i reach for dynamic programming", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    ("whats the difference between a stack and a heap data structure", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("explain NP-complete, is it just really hard problems", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("how does a hash table handle collisions", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("i want to write a tiny compiler, what are the phases", [("cs-theory", 1.0)], ["cs-theory/compilers"]),
    ("whats a balanced tree and why do i care", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("can you explain big O without the scary math", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("greedy vs dynamic programming, how do i know which", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    # math
    ("i forgot how to take a derivative of x^3, help", [("math", 1.0)], ["math/calculus"]),
    ("whats an eigenvalue in plain english", [("math", 1.0)], ["math/linear-algebra"]),
    ("solve for x: 2x + 3 = 11, and show steps pls", [("math", 1.0)], ["math/algebra"]),
    ("explain bayes theorem with an example i can follow", [("math", 1.0)], ["math/probability"]),
    ("whats modular arithmetic, the clock thing?", [("math", 1.0)], ["math/number-theory"]),
    ("how do i integrate by parts again", [("math", 1.0)], ["math/calculus"]),
    ("whats the difference between permutations and combinations", [("math", 1.0)], ["math/combinatorics"]),
    ("why is matrix multiplication not commutative", [("math", 1.0)], ["math/linear-algebra"]),
    # science
    ("why do leaves change color in fall, is it the chlorophyll", [("science", 1.0)], ["science/biology"]),
    ("whats the difference between mitosis and meiosis again", [("science", 1.0)], ["science/biology"]),
    ("explain why ice floats, shouldnt solids sink", [("science", 1.0)], ["science/chemistry"]),
    ("how does a catalyst speed up a reaction without being used up", [("science", 1.0)], ["science/chemistry"]),
    ("whats entropy, everyone says disorder but that feels vague", [("science", 1.0)], ["science/physics"]),
    ("why do we feel weightless on a roller coaster drop", [("science", 1.0)], ["science/physics"]),
    ("how do plate tectonics cause earthquakes", [("science", 1.0)], ["science/earth-space"]),
    ("whats a black hole, like what actually is it", [("science", 1.0)], ["science/earth-space"]),
    ("how does DNA actually store information", [("science", 1.0)], ["science/biology"]),
    ("why is the ocean salty but rivers arent", [("science", 1.0)], ["science/earth-space"]),
    ("whats pH and why does it matter for my fish tank", [("science", 1.0)], ["science/chemistry"]),
    ("how does natural selection actually work over time", [("science", 1.0)], ["science/biology"]),
    # cyber-offensive
    ("found a login form, how would i test it for sql injection", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("whats the first thing you do on a pentest, like recon-wise", [("cyber-offensive", 1.0)], ["cyber-offensive/recon"]),
    ("explain a buffer overflow, im new to binary exploitation", [("cyber-offensive", 1.0)], ["cyber-offensive/binary-exploitation"]),
    ("got a low priv shell, how do i escalate to root on linux", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("whats a reverse shell and why does everyone use nc for it", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("how do i enumerate open ports without being too loud", [("cyber-offensive", 1.0)], ["cyber-offensive/scanning"]),
    ("whats SSRF and how is it different from CSRF", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    # cyber-defensive
    ("my SIEM is drowning in alerts, how do i tune it down", [("cyber-defensive", 1.0)], ["cyber-defensive/siem"]),
    ("how do i write a detection rule for suspicious powershell", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("we got breached, whats the first step of incident response", [("cyber-defensive", 1.0)], ["cyber-defensive/incident-response"]),
    ("how do i do memory forensics on a possibly infected box", [("cyber-defensive", 1.0)], ["cyber-defensive/forensics"]),
    ("whats MITRE ATT&CK and how do i use it for detection", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("how do i tell if this binary is malware without running it", [("cyber-defensive", 1.0)], ["cyber-defensive/malware-analysis"]),
    ("whats threat hunting vs just waiting for alerts", [("cyber-defensive", 1.0)], ["cyber-defensive/threat-intel"]),
    ("how do i harden a fresh ubuntu server", [("cyber-defensive", 1.0)], ["cyber-defensive/hardening"]),
    # cyber-appsec
    ("how do i fuzz a parser, whats a good harness look like", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("whats SAST and does it actually catch real bugs", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("reviewing code for security, what do i look for first", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("found a weird crash, how do i tell if its exploitable", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("whats the difference between sast and dast", [("cyber-appsec", 1.0)], ["cyber-appsec/dast"]),
    ("how do i stop injection bugs in my web app", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    # cyber-netsec-crypto
    ("is it ok to use AES-ECB, a tutorial told me to", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("explain how RSA works at a high level", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("whats a padding oracle attack", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptanalysis"]),
    ("how do certificates and CAs actually establish trust", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/pki"]),
    ("how do i read a pcap to find whats talking to what", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/packet-analysis"]),
    ("whats the difference between hashing and encryption, people mix these up", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    # cyber-specialized
    ("how do i start looking at ICS/SCADA security, its all new to me", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats OSINT and how do i gather it on a target legally", [("cyber-specialized", 1.0)], ["cyber-specialized/osint"]),
    ("my s3 bucket is public, how bad is that and how do i check others", [("cyber-specialized", 1.0)], ["cyber-specialized/cloud-sec"]),
    ("how do i analyze an apk for sketchy behavior", [("cyber-specialized", 1.0)], ["cyber-specialized/mobile-sec"]),
    ("whats a good phishing awareness test for employees", [("cyber-specialized", 1.0)], ["cyber-specialized/social-engineering"]),
    ("how do iot devices usually get hacked", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats an evil twin attack on wifi", [("cyber-specialized", 1.0)], ["cyber-specialized/wireless-sec"]),
    # cyber-data
    ("how do i find brute force attempts in a pile of auth logs", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("whats a good way to detect anomalies in network traffic", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    ("can i train a model to spot phishing urls", [("cyber-data", 1.0)], ["cyber-data/ml-for-security"]),
    ("how would i build an attack graph from my logs", [("cyber-data", 1.0)], ["cyber-data/security-graph-analytics"]),
    ("whats beaconing and how do i spot it in netflow", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("isolation forest for security, does that actually work", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
]

# Second batch to bring every category to >=15 short prompts. Same messy style.
P += [
    # coding
    ("ruby: whats the difference between a block, proc and lambda", [("coding", 1.0)], ["coding/ruby"]),
    ("swift optionals keep crashing my app, how do i unwrap safely", [("coding", 1.0)], ["coding/swift"]),
    ("kotlin coroutines vs threads, when do i use which", [("coding", 1.0)], ["coding/kotlin"]),
    ("whats the fastest way to sort a huge file in sql", [("coding", 1.0)], ["coding/sql"]),
    ("hand-writing some x86 asm, how do i do a function call", [("coding", 1.0)], ["coding/asm"]),
    # systems
    ("windows: how do i see what dll a process loaded", [("systems", 1.0)], ["systems/windows-internals"]),
    ("whats an rtos and do i need one for my robot", [("systems", 1.0)], ["systems/rtos"]),
    ("my embedded firmware bricks the board on update, help", [("systems", 1.0)], ["systems/firmware"]),
    ("how do interrupts actually work at the hardware level", [("systems", 1.0)], ["systems/kernel"]),
    ("whats copy-on-write memory and when does it kick in", [("systems", 1.0)], ["systems/linux"]),
    ("how do i write a driver that talks to a usb device", [("systems", 1.0)], ["systems/drivers"]),
    ("atomic ops vs mutex, whats cheaper", [("systems", 1.0)], ["systems/concurrency"]),
    # networking
    ("whats the difference between tcp and udp in practice", [("networking", 1.0)], ["networking/tcp-ip"]),
    ("my dns keeps resolving to an old ip, caching issue?", [("networking", 1.0)], ["networking/dns"]),
    ("how does https actually keep my traffic private", [("networking", 1.0)], ["networking/tls"]),
    ("whats a vlan and why would i segment my network", [("networking", 1.0)], ["networking/sdn"]),
    ("explain nat, i dont get how it maps ports", [("networking", 1.0)], ["networking/protocols"]),
    ("5ghz vs 2.4ghz wifi, which should i use", [("networking", 1.0)], ["networking/wireless"]),
    ("how do routers decide the best path", [("networking", 1.0)], ["networking/routing-bgp"]),
    # cloud
    ("whats IAM and why is everyone obsessed with least privilege", [("cloud", 1.0)], ["cloud/aws"]),
    ("my docker image is 2gb, how do i make it smaller", [("cloud", 1.0)], ["cloud/docker"]),
    ("how do i monitor a k8s cluster, whats the stack", [("cloud", 1.0)], ["cloud/observability"]),
    ("terraform state is corrupted, can i recover it", [("cloud", 1.0)], ["cloud/terraform-iac"]),
    ("azure or aws for machine learning workloads", [("cloud", 1.0)], ["cloud/azure"]),
    ("whats a service mesh and do i need one", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("how do i set up blue green deployment", [("cloud", 1.0)], ["cloud/ci-cd"]),
    # data
    ("whats the n+1 query problem and how do i fix it", [("data", 1.0)], ["data/databases"]),
    ("when should i use nosql over postgres", [("data", 1.0)], ["data/databases"]),
    ("how does change data capture work", [("data", 1.0)], ["data/data-engineering"]),
    ("whats a materialized view and when is it worth it", [("data", 1.0)], ["data/databases"]),
    ("spark job is super slow, how do i debug it", [("data", 1.0)], ["data/data-engineering"]),
    ("whats eventual consistency and why does it scare people", [("data", 1.0)], ["data/distributed-systems"]),
    ("how do i do exactly-once processing in a stream", [("data", 1.0)], ["data/streaming"]),
    # web-mobile
    ("why is my react app so slow on first load", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("whats CORS and why does it keep blocking my requests", [("web-mobile", 1.0)], ["web-mobile/backend"]),
    ("how do i version a public api without breaking clients", [("web-mobile", 1.0)], ["web-mobile/api-design"]),
    ("swiftui vs uikit for a new ios app", [("web-mobile", 1.0)], ["web-mobile/ios"]),
    ("my android app drains battery, how do i profile it", [("web-mobile", 1.0)], ["web-mobile/android"]),
    ("whats server side rendering and do i need it", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("how do browsers actually render a page", [("web-mobile", 1.0)], ["web-mobile/browsers"]),
    # ai-ml
    ("whats the difference between bagging and boosting", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("how do i pick the learning rate, its all guesswork", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("whats a good loss for imbalanced classification", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    ("how does beam search work in decoding", [("ai-ml", 1.0)], ["ai-ml/nlp"]),
    ("what is data augmentation for images", [("ai-ml", 1.0)], ["ai-ml/cv"]),
    ("whats the exploration exploitation tradeoff in RL", [("ai-ml", 1.0)], ["ai-ml/rl"]),
    # cs-theory
    ("whats a trie and when is it better than a hashmap", [("cs-theory", 1.0)], ["cs-theory/data-structures"]),
    ("explain amortized analysis, the dynamic array thing", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("how does a parser turn text into a syntax tree", [("cs-theory", 1.0)], ["cs-theory/compilers"]),
    ("whats a context free grammar", [("cs-theory", 1.0)], ["cs-theory/formal-methods"]),
    ("when is bfs better than dfs", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    ("whats the halting problem, is it really unsolvable", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    # math
    ("whats a taylor series for, intuitively", [("math", 1.0)], ["math/calculus"]),
    ("how do i find the determinant of a 3x3 matrix", [("math", 1.0)], ["math/linear-algebra"]),
    ("explain standard deviation vs variance", [("math", 1.0)], ["math/statistics"]),
    ("whats gradient descent doing geometrically", [("math", 1.0)], ["math/optimization"]),
    ("how do i prove something by induction", [("math", 1.0)], ["math/logic"]),
    ("whats a graph in the math sense, not a chart", [("math", 1.0)], ["math/graph-theory"]),
    ("solve this: a train leaves at 60mph...", [("math", 1.0)], ["math/word-problems"]),
    # science
    ("how do vaccines teach the immune system", [("science", 1.0)], ["science/biology"]),
    ("whats the greenhouse effect, simply", [("science", 1.0)], ["science/earth-space"]),
    ("why do metals conduct electricity", [("science", 1.0)], ["science/physics"]),
    # cyber-offensive
    ("whats privilege escalation on windows look like", [("cyber-offensive", 1.0)], ["cyber-offensive/privesc"]),
    ("how do i find a zero day, like whats the process", [("cyber-offensive", 1.0)], ["cyber-offensive/zeroday"]),
    ("whats lateral movement in an AD environment", [("cyber-offensive", 1.0)], ["cyber-offensive/lateral-movement"]),
    ("how do i set up a c2 for a red team op", [("cyber-offensive", 1.0)], ["cyber-offensive/c2"]),
    ("explain exploit development basics", [("cyber-offensive", 1.0)], ["cyber-offensive/exploit-dev"]),
    ("how do i enumerate a target after initial access", [("cyber-offensive", 1.0)], ["cyber-offensive/enumeration"]),
    ("whats an xxe attack", [("cyber-offensive", 1.0)], ["cyber-offensive/web-exploitation"]),
    ("how do i scan a /16 quickly", [("cyber-offensive", 1.0)], ["cyber-offensive/scanning"]),
    # cyber-defensive
    ("whats blue team vs purple team", [("cyber-defensive", 1.0)], ["cyber-defensive/blue-team"]),
    ("how do i build a threat intel program from scratch", [("cyber-defensive", 1.0)], ["cyber-defensive/threat-intel"]),
    ("whats sysmon and should i deploy it everywhere", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("how do i do disk forensics on a windows machine", [("cyber-defensive", 1.0)], ["cyber-defensive/forensics"]),
    ("write a yara rule to catch a known malware family", [("cyber-defensive", 1.0)], ["cyber-defensive/malware-analysis"]),
    ("whats the containment phase in IR", [("cyber-defensive", 1.0)], ["cyber-defensive/incident-response"]),
    ("how do i reverse a stripped binary", [("cyber-defensive", 1.0)], ["cyber-defensive/reverse-engineering"]),
    # cyber-appsec
    ("whats dast and how is it different from sast", [("cyber-appsec", 1.0)], ["cyber-appsec/dast"]),
    ("how do i add sanitizers to my fuzzing build", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("whats taint analysis in static scanning", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("how do i triage 500 sast findings", [("cyber-appsec", 1.0)], ["cyber-appsec/sast"]),
    ("whats a secure code review checklist look like", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how do i tell a real vuln from a false positive", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("whats dynamic testing with a proxy like burp", [("cyber-appsec", 1.0)], ["cyber-appsec/dast"]),
    # cyber-netsec-crypto
    ("whats an ids vs an ips", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/ids-ips"]),
    ("how do i write a firewall rule to block a subnet", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/firewalls"]),
    ("whats perfect forward secrecy", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how do i analyze a pcap for exfil", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/packet-analysis"]),
    ("whats a nonce and why does reuse break things", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how does certificate revocation work, ocsp vs crl", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/pki"]),
    ("whats a known plaintext attack", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptanalysis"]),
    # cyber-specialized
    ("how do modbus and other ICS protocols get attacked", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
    ("whats a good methodology for a cloud pentest", [("cyber-specialized", 1.0)], ["cyber-specialized/methodology"]),
    ("how do i do recon with just osint, no scanning", [("cyber-specialized", 1.0)], ["cyber-specialized/osint"]),
    ("whats certificate pinning and how do i bypass it on mobile", [("cyber-specialized", 1.0)], ["cyber-specialized/mobile-sec"]),
    ("how do i deauth a wifi client for testing", [("cyber-specialized", 1.0)], ["cyber-specialized/wireless-sec"]),
    ("whats a pretext in social engineering", [("cyber-specialized", 1.0)], ["cyber-specialized/social-engineering"]),
    ("how do i research a new attack technique properly", [("cyber-specialized", 1.0)], ["cyber-specialized/research"]),
    # cyber-data
    ("how do i baseline normal behavior to spot anomalies", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    ("whats entity resolution in security graphs", [("cyber-data", 1.0)], ["cyber-data/security-graph-analytics"]),
    ("how do i extract features from a binary for ml", [("cyber-data", 1.0)], ["cyber-data/ml-for-security"]),
    ("how do i parse and correlate windows event logs at scale", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("can graph analytics find lateral movement", [("cyber-data", 1.0)], ["cyber-data/security-graph-analytics"]),
    ("whats a good model for detecting dga domains", [("cyber-data", 1.0)], ["cyber-data/ml-for-security"]),
    ("how do i reduce false positives in anomaly detection", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
]

P += [
    ("whats fuzzing with libfuzzer actually like to set up", [("cyber-appsec", 1.0)], ["cyber-appsec/fuzzing"]),
    ("how do i audit a jwt implementation for bugs", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    ("how do i cluster similar security alerts to cut noise", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    ("whats a good query to find data exfil in proxy logs", [("cyber-data", 1.0)], ["cyber-data/log-analysis"]),
    ("whats diffie hellman key exchange, simply", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how do i spot a weak or outdated tls config", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("how would i test a smart lock for vulnerabilities", [("cyber-specialized", 1.0)], ["cyber-specialized/iot-ics-scada"]),
]

COMBOS: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    ("write a lil python script that scans a host for open ports and prints banners", [("coding", 0.45), ("cyber-offensive", 0.35), ("networking", 0.2)], ["coding/python", "cyber-offensive/recon"]),
    ("i wanna train a model to flag weird logins from our auth logs", [("ai-ml", 0.4), ("cyber-data", 0.4), ("data", 0.2)], ["ai-ml/ml", "cyber-data/anomaly-detection"]),
    ("deploy my go api on k8s with proper tls, where do i start", [("coding", 0.3), ("cloud", 0.4), ("cyber-netsec-crypto", 0.3)], ["coding/go", "cloud/kubernetes", "cyber-netsec-crypto/pki"]),
    ("red-black tree in rust, from scratch, no libs", [("cs-theory", 0.5), ("coding", 0.5)], ["cs-theory/data-structures", "coding/rust"]),
    ("fuzz my C parser and help me figure out if the crash is exploitable", [("cyber-appsec", 0.5), ("cyber-offensive", 0.3), ("coding", 0.2)], ["cyber-appsec/fuzzing", "cyber-offensive/binary-exploitation"]),
    ("sql query is slow AND i think someones probing the db, two problems", [("data", 0.5), ("cyber-defensive", 0.5)], ["data/databases", "cyber-defensive/detection"]),
    ("add csrf protection to my django app pls", [("web-mobile", 0.5), ("cyber-appsec", 0.5)], ["web-mobile/backend", "cyber-appsec/code-audit"]),
    ("capture the tls session and decrypt it for analysis", [("cyber-netsec-crypto", 0.55), ("networking", 0.45)], ["cyber-netsec-crypto/packet-analysis", "networking/tls"]),
    ("whats the complexity of training a transformer, math-wise", [("cs-theory", 0.5), ("ai-ml", 0.5)], ["cs-theory/complexity", "ai-ml/deep-learning"]),
    ("reverse engineer this firmware blob and script the unpacking in python", [("cyber-specialized", 0.4), ("cyber-defensive", 0.3), ("coding", 0.3)], ["cyber-specialized/iot-ics-scada", "cyber-defensive/reverse-engineering"]),
    ("implement aes-gcm in rust, need it for a file encryptor", [("cyber-netsec-crypto", 0.55), ("coding", 0.45)], ["cyber-netsec-crypto/cryptography", "coding/rust"]),
    ("build a feature pipeline from postgres for a recommender model", [("data", 0.5), ("ai-ml", 0.5)], ["data/data-engineering", "ai-ml/ml"]),
    ("containerize this linux daemon and ship it to k8s", [("systems", 0.35), ("cloud", 0.45), ("coding", 0.2)], ["systems/linux", "cloud/kubernetes"]),
    ("write a suricata rule for dns tunneling pls", [("cyber-defensive", 0.6), ("networking", 0.4)], ["cyber-defensive/detection", "networking/dns"]),
    ("memory allocator in C, and explain the complexity", [("systems", 0.4), ("coding", 0.35), ("cs-theory", 0.25)], ["systems/linux", "coding/c"]),
    ("prove gradient descent converges for convex loss, with the math", [("math", 0.55), ("ai-ml", 0.45)], ["math/analysis", "ai-ml/deep-learning"]),
    ("find an ssrf in this node api and write a poc", [("cyber-offensive", 0.55), ("web-mobile", 0.45)], ["cyber-offensive/web-exploitation", "web-mobile/backend"]),
    ("tune tcp buffers on a busy linux box for throughput", [("networking", 0.5), ("systems", 0.5)], ["networking/tcp-ip", "systems/linux"]),
    ("train an isolation forest on netflow to catch intrusions", [("cyber-data", 0.55), ("ai-ml", 0.45)], ["cyber-data/ml-for-security", "ai-ml/ml"]),
    ("set up a vpc with private subnets and a nat gateway on aws", [("cloud", 0.6), ("networking", 0.4)], ["cloud/aws", "networking/tcp-ip"]),
    ("crack this captured wpa2 handshake and tell me whats weak", [("cyber-specialized", 0.5), ("cyber-netsec-crypto", 0.5)], ["cyber-specialized/wireless-sec", "cyber-netsec-crypto/cryptanalysis"]),
    ("serve a pytorch model behind fastapi reading from redis", [("ai-ml", 0.4), ("coding", 0.3), ("data", 0.3)], ["ai-ml/mlops", "coding/python", "data/databases"]),
]

OOS: list[str] = [
    "whats a good banana bread recipe, mine always comes out dense",
    "who actually won the 2018 world cup, i keep forgetting",
    "recommend me a comedy to watch tonight im so tired",
    "is it gonna rain in lisbon this weekend?",
    "how do i get red wine out of a white shirt ughhh",
    "what should i get my sister for her 30th",
    "plan me a chill weekend in rome",
    "tell me a joke about cats",
    "write a short poem about the rain pls",
    "how long does it take to train for a 5k from zero",
    "fun fact about octopuses? my kid asked and i blanked",
    "is matcha actually healthier than coffee or is that hype",
    "best way to keep basil alive, mine keeps dying",
    "whats a thoughtful housewarming gift",
    "how do i fold a fitted sheet, this is impossible",
    "recommend a podcast for a long drive",
    "whats a good board game for 4 adults",
    "how do i remove a coffee stain from a mug",
    "suggest a hobby thats cheap to start",
    "whats the best way to meal prep for the week",
    "write me a birthday toast for my dad",
    "any tips for sleeping better, i keep waking at 3am",
    "whats a fun fact about the ocean",
    "how do i keep squirrels out of my garden",
    "whats a quick healthy breakfast i can make in 5 min",
    "recommend a feel-good movie for a rainy sunday",
    "how do i get my toddler to eat vegetables",
    "whats the best way to learn to swim as an adult",
    "any good audiobooks for a long flight",
    "how do i clean cast iron without ruining it",
    "whats a nice day trip from amsterdam",
    "tell me a dad joke",
    "how do i keep cut flowers fresh longer",
    "whats a good playlist for working out",
    "how do i fix a squeaky door hinge",
    "whats the difference between espresso and drip coffee",
    "recommend a mystery novel please",
    "how do i start composting at home",
    "whats a fun indoor activity for a rainy day with kids",
    "how do i remove chewing gum from hair",
    "whats a good gift for a wedding under 50 bucks",
    "how long should i boil an egg for soft yolk",
    "any tips for staying awake on a night drive",
    "whats a beginner friendly houseplant",
    "how do i make my coffee less bitter",
]


def slug(t: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", t.strip().lower()).encode()).hexdigest()[:12]


def texts_of(path: Path) -> set[str]:
    out: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.add(slug(json.loads(line)["text"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent.parent.parent / "nina-data" / "field-corpora"
    ap.add_argument("--out", type=Path, default=base / "router-eval-v2.jsonl")
    ap.add_argument("--train", type=Path, default=base / "router-train-prompts.jsonl")
    ap.add_argument("--v1", type=Path, default=base / "router-eval.jsonl")
    args = ap.parse_args()

    avoid = texts_of(args.train) | texts_of(args.v1)  # keep independent of train + v1
    seen = set(avoid)
    lines, counts = [], {}
    n = 0

    def add(text, cats, subs, style):
        nonlocal n
        s = slug(text)
        if s in seen:
            return
        seen.add(s)
        total = sum(w for _, w in cats) or 1.0
        categories = [{"id": c, "weight": round(w / total, 3)} for c, w in cats]
        lines.append(json.dumps({"id": f"v2-{style}-{n:04d}", "text": text,
                                 "categories": categories, "subfields": subs, "style": style}, ensure_ascii=False))
        n += 1
        for c, _ in cats:
            counts[c] = counts.get(c, 0) + 1

    for text, cats, subs in P:
        add(text, cats, subs, "short")
    for text, cats, subs in COMBOS:
        add(text, cats, subs, "combo")
    for text in OOS:
        add(text, [], [], "oos")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from collections import Counter
    st = Counter(json.loads(x)["style"] for x in lines)
    total = len(lines)
    print(f"router-eval-v2: {total} ({st['short']} short, {st['combo']} combo, {st['oos']} oos "
          f"= {round(100 * st['oos'] / total)}%) -> {args.out}")
    print(f"per-category (short): {dict(sorted((k, v) for k, v in counts.items()))}")
    print(f"deduped vs train+v1 ({len(avoid)} texts): independent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
