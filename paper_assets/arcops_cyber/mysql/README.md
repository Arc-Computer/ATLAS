# SecRL MySQL Runtime Setup

This directory stores the auto-generated MySQL bootstrap used to mirror Microsoft's SecRL ArcOps-Cyber benchmark. Follow the steps below to start the incidents locally and expose them to the ATLAS runtime.

## 1. Regenerate the init script (optional)

```bash
cd /Users/jarrodbarnes/ATLAS
python scripts/arcops_cyber/generate_secrl_sql.py
```

This scans `paper_assets/arcops_cyber/data/data_anonymized` and rewrites `init.sql` with `CREATE DATABASE`, `LOAD DATA INFILE`, and the read-only `atlas` user grants.

## 2. Launch the SecRL MySQL container

```bash
cd /Users/jarrodbarnes/ATLAS
docker run \
  --name atlas-secrl-mysql \
  --env MYSQL_ROOT_PASSWORD=admin \
  --publish 3307:3306 \
  --mount type=bind,src="$PWD/paper_assets/arcops_cyber/mysql",dst=/docker-entrypoint-initdb.d,readonly \
  --mount type=bind,src="$PWD/paper_assets/arcops_cyber/data",dst=/var/lib/mysql-files,readonly \
  mysql:9.0 --secure-file-priv=/var/lib/mysql-files --local-infile=1
```

The entrypoint executes `init.sql`, materialising the eight incident databases (`incident_5`, `incident_34`, …, `incident_322`) and granting the read-only `atlas` user (password `atlas`).

> **Tip:** Use `docker logs -f atlas-secrl-mysql` to watch for `Ready for connections`. Subsequent restarts can omit the bind mounts once the data directory is persisted.

## 3. Export runtime environment variables

Add the following to your shell profile or `.env`:

```bash
export ATLAS_SECRL_HOST=127.0.0.1
export ATLAS_SECRL_PORT=3307
export ATLAS_SECRL_USER=atlas
export ATLAS_SECRL_PASSWORD=atlas
```

The SecRL SQL adapter reads these variables on first use.

## 4. Smoke-test the tool adapter

With the container running and environment variables exported:

```bash
cd /Users/jarrodbarnes/ATLAS
python - <<'PY'
import asyncio, json
from atlas_core.tools import secrl_sql_adapter

async def main():
    payload = {
        "name": "process_owner_lookup",
        "arguments": {
            "incident_id": "incident_38",
            "host": "vnevado-win11a",
            "process_id": "8932",
        },
    }
    result = await secrl_sql_adapter.student_adapter("", metadata={"tool": payload})
    print(json.dumps(json.loads(result), indent=2))

asyncio.run(main())
PY
```

You should see a single `DeviceProcessEvents` row whose `AccountName` field is `nathans`, matching the ArcOps benchmark ground truth.
