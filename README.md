# duckdb-milvus-backup

用 SQL 直接读取 [Milvus](https://github.com/milvus-io/milvus) 备份的 DuckDB 扩展——在把备份恢复进一个运行中的 Milvus 实例之前,先检查里面的 collection、partition 和 binlog 数据。

仓库初始化自实验性的 [duckdb/extension-template-rs](https://github.com/duckdb/extension-template-rs) 模板:纯 Rust 扩展,基于 DuckDB C Extension API v2(`duckdb-neo` + `duckdb-loadable-macros`,目标 DuckDB **v2.0.0**,版本必须精确匹配),无需构建 DuckDB 本体、无需 C/C++ 代码,`extension-ci-tools` 的 CI/CD 链路已预配置。

## 路线图

| 阶段 | 函数 | 状态 | 说明 |
|---|---|---|---|
| 0 | `milvus_backup_version()` | ✅ 已实现 | 扩展版本号,编译期烤入;同时打通 build/load/test 全管线 |
| 1 | `milvus_backups(root_path)` | 🚧 计划中 | 备份清单:一备份一行,平铺标量列(name、id、milvus_version、created_at、size_bytes、collection_count……)。只读 meta JSON,不碰 binlog |
| 2 | `milvus_backup_collections(root_path, backup_name)` | 🚧 计划中 | 一 collection 一行,供 WHERE/ORDER BY/JOIN;schema 等二级信息走显式选列 |
| 3 | `milvus_backup_scan(root_path, backup_name, collection)` | 🚧 计划中 | 扫描 binlog 数据。`claim_batch` 按文件并行、projection/filter pushdown、vector 列用 `FLOAT[dim]` 定长数组 |
| 远期 | 多 collection 扫描(schema 一致性校验)、S3 支持(具名参数传 endpoint/credential)、`milvus_backup_verify()` | 💡 | |

设计原则:`SELECT *` 应能回答"这个备份可信吗、能用吗";凡是从 meta 里无法确知的信息(最终行数、完整性)给 NULL 或布尔值,不给伪造的 0/成功;指标类函数绝不读 binlog(整个桶也要秒回)。

## 使用

用 `-unsigned` 启动 duckdb,按文件路径加载扩展(路径必须为绝对路径,CLI 版本须精确匹配 v2.0.0):

```sh
./build/duckdb -unsigned
```

```sql
LOAD '/abs/path/to/build/debug/extension/milvus_backup/milvus_backup.duckdb_extension';

SELECT milvus_backup_version();
```

```
┌─────────────────────────┐
│ milvus_backup_version() │
│         varchar         │
├─────────────────────────┤
│ milvus_backup 0.1.0     │
└─────────────────────────┘
```

以下为规划中的接口形态(尚未实现):

```sql
SELECT * FROM milvus_backups('/mnt/backups');
SELECT * FROM milvus_backup_collections('/mnt/backups', 'backup_20261027');
SELECT * FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', collection = 'messages');
```

## 项目结构

Cargo workspace 三个 crate:

- `crates/core`(`milvus_backup_core`):函数实现与注册,全部扩展逻辑在这里。
  约定:**一个函数一个文件**(`version.rs`,将来的 `backups.rs`、`collections.rs`……),每个文件自带 `pub(crate) fn register`,`lib.rs` 只有模块声明和顶层 `register`。
- `crates/native`(`milvus_backup`):native `cdylib`,DuckDB 扩展入口。
- `crates/wasm`(`milvus_backup_wasm`):Wasm `staticlib`,同一入口。

依赖版本在根 `Cargo.toml` 统一管理;`upgrade.sh` 同步升级 duckdb-neo / duckdb-loadable-macros / DuckDB 版本。

## 构建

依赖:Rust 工具链、Python3(带 venv)、Make、Git。

```shell
git clone --recurse-submodules <repo-url>
make configure   # 建 Python venv(装 DuckDB 测试器),探测平台
make debug       # cargo 构建 + 追加扩展元数据,产物在 build/debug
make release     # 优化构建
```

### WASM

`make wasm_eh` 生成 `./build/wasm_eh/release/milvus_backup.duckdb_extension.wasm`,可在 Node.js 或浏览器中使用。

## 测试

SQLLogicTest 格式,位于 `test/sql/`,通过 `make configure` 安装的 DuckDB Python 客户端运行:

```shell
make test_debug    # 或 make test_release
```

## 调试

已配置 VS Code 断点调试(`.vscode/`):安装 CodeLLDB 插件后直接 **F5**,选 "run version query" 自动 LOAD 并命中断点,或选 "interactive" 在集成终端里手动敲 SQL。

macOS 注意事项:

- `build/duckdb` 官方二进制带 Hardened Runtime,调试器无法附加,需重签为 ad-hoc:`codesign --force --sign - ./build/duckdb`(重新下载 CLI 后需重做);
- 若 `make debug` 后 LOAD 被 SIGKILL(exit 137),先 `rm` 掉旧扩展产物再构建(inode 复用导致的签名毒页问题);
- DuckDB 内部无符号,只能调试扩展自身代码;多线程下断点乱跳时用 `SET threads TO 1`。

### 版本切换

```shell
make clean_all
DUCKDB_TEST_VERSION=v1.3.2 make configure
make debug
make test_debug
```
