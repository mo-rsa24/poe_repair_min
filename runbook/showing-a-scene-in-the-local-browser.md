# Showing a scene in the local browser

Navigation: 📋 [Index](00-INDEX.md)

Contents: [1. Serve the scenes folder from a tmux session](#1-serve-the-scenes-folder-from-a-tmux-session) |
[2. Forward the port and open it on the laptop](#2-forward-the-port-and-open-it-on-the-laptop)

The static scenes under `artifacts/scenes/` (single `index.html` files, no build) are shown by
serving that folder once on the session node and tunnelling the port from the laptop. Scenes
that are Vite apps (`logsnr-explainer`, `how-much-correction-is-needed`) have their own `npm run
dev` blocks in their READMEs; this theme is for everything that is just a file.

## 1. Serve the scenes folder from a tmux session

`ran 2026-08-29`

Navigation: 📋 [TOC](#showing-a-scene-in-the-local-browser) | [Next](#2-forward-the-port-and-open-it-on-the-laptop) ➡️

**When you need this**

A scene exists on the cluster and you want it viewable in a browser without copying files down.

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| port | `8800` | any free port; `ss -ltn \| grep 8800` prints nothing when free |
| session name | `scenes` | your choice; `scenes` is what the rest of this theme assumes |

```bash
tmux new-session -d -s scenes 'cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/scenes && python3 -m http.server 8800 --bind 127.0.0.1'
tmux ls
curl -sI http://localhost:8800/noise-shell-and-basins/index.html | head -3
```

✅ **The session is listed and the file answers 200.** Real output, 2026-08-29, mscluster85:

```
scenes: 1 windows (created Sat Aug 29 05:50:56 2026)
HTTP/1.0 200 OK
Server: SimpleHTTP/0.6 Python/3.12.10
Date: Sat, 29 Aug 2026 05:50:57 GMT
```

❌ **`Address already in use` in the tmux pane** (see it with `tmux attach -t scenes`): another
server holds the port. `ss -ltnp | grep 8800` names the process; pick another port and change it
in both this recipe and the forward in [recipe 2](#2-forward-the-port-and-open-it-on-the-laptop).

**Variations**

- Watch the request log live: `tmux attach -t scenes`, detach with `Ctrl-b` then `d`.
- Stop the server: `tmux kill-session -t scenes`.
- Serve one scene only instead of all of them: change the `cd` target to that scene's folder;
  the URL in recipe 2 then drops the scene-name segment.

<details>
<summary><b>Why bind to 127.0.0.1, and why tmux rather than nohup</b></summary>

**The bind**

Bound to localhost, the server is unreachable from the rest of the shared cluster; the only way
in is the SSH tunnel, which is the access model you want on a node other people use. It also
means no port collision with anyone serving on the same number bound elsewhere.

**The tmux part**

`nohup` is the house pattern for long runs across many settings because they must survive the
session; a scene server is
the opposite, something you want to find, watch, and kill by name a week later. `tmux ls` finds
it, attach shows its request log, kill-session ends it cleanly.

</details>

## 2. Forward the port and open it on the laptop

`adapted 2026-08-29`

Navigation: ⬅️ [1. Serve the scenes folder from a tmux session](#1-serve-the-scenes-folder-from-a-tmux-session) | 📋 [TOC](#showing-a-scene-in-the-local-browser)

**When you need this**

Recipe 1 is serving on the node and you want the page in your local browser.

**Needs first**

The server from [recipe 1](#1-serve-the-scenes-folder-from-a-tmux-session), still running
(`tmux ls` on the node shows `scenes`).

**Fill in**

| What | Example | Where to get it |
|---|---|---|
| node | `mscluster85.ms.wits.ac.za` | the current session node, per [environment/overview.md](../environment/overview.md); changes when the session moves |
| port | `8800` | the same port recipe 1 used |

In a terminal **on the laptop**, not on the cluster:

```bash
ssh -L 8800:localhost:8800 mmolefe@mscluster85.ms.wits.ac.za
```

Leave that terminal open (the tunnel lives as long as the SSH session), then browse to:

```
http://localhost:8800/noise-shell-and-basins/
```

✅ **The scene renders in the local browser.** The folder root (`http://localhost:8800/`) lists
every scene directory; equations need the laptop's internet because KaTeX loads from a CDN, and
everything else is served through the tunnel.

❌ **Browser says connection refused**: the tunnel is up but nothing is listening on the node,
so the server died or was never started; go to
[recipe 1](#1-serve-the-scenes-folder-from-a-tmux-session).
❌ **SSH itself hangs or refuses**: wrong node name; check the current session node in
[environment/overview.md](../environment/overview.md).

**Variations**

- Local port 8800 already taken on the laptop: `ssh -L 9900:localhost:8800 ...` and browse
  `http://localhost:9900/...`.
- Tunnel without an interactive shell: add `-N` to the ssh command; `Ctrl-c` closes it.

<details>
<summary><b>Is this reverse port forwarding?</b></summary>

**The name versus the flag**

The house name for this move is reverse port forwarding, and the handoff rule is that any app
served on a cluster node includes this command, not just the URL. In ssh's own vocabulary `-L`
is local forwarding (laptop listens, traffic goes to the node); `-R` is what ssh calls reverse.
The command above is the correct one either way; only the naming differs.

</details>
