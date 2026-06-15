import sys, os, json, urllib.request, urllib.parse
import numpy as np

BASE = 'http://127.0.0.1:5001'


def get(path):
    try:
        with urllib.request.urlopen(BASE + path) as r:
            return json.loads(r.read().decode()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def check(cond, msg):
    sym = '✅' if cond else '❌'
    print(f"  {sym} {msg}")
    return cond


def main():
    all_ok = True

    section("0. POST /api/data/refresh (先重置到清洁数据)")
    data, code = post('/api/data/refresh', {'dirty_ratio': 0.0})
    all_ok &= check(code == 200 and data.get('status') == 'data_refreshed',
                    f"重置到 clean 数据 OK (code={code})")

    section("1. GET /api/health (清洁数据)")
    data, code = get('/api/health')
    all_ok &= check(code == 200, f"HTTP 200 (got {code})")
    all_ok &= check(data.get('status') == 'healthy', f"status=healthy (got {data.get('status')})")
    all_ok &= check(data.get('trained_machines') == 4, f"4 machines trained")

    section("2. GET /api/data-quality (清洁数据)")
    data, code = get('/api/data-quality')
    all_ok &= check(code == 200, f"HTTP 200")
    reports = data.get('reports', {})
    all_ok &= check(len(reports) == 4, f"4 quality reports (got {len(reports)})")
    for mid, rep in reports.items():
        qw = rep.get('quality_warning', '?')
        nr = rep.get('null_ratio', 0)
        # 注意：即使"清洁"数据也会有模拟的约 0%~5% 随机缺失，放宽阈值
        all_ok &= check(nr < 0.12, f"{mid} 空值率={nr:.1%} < 12%")

    section("3. POST /api/machines/<id>/optimize (非法参数)")
    bad_body = {
        'vibration_freq': None,
        'inclination_angle': 'bad_value',
        'fan_airflow': -9999,
        'amplitude': 1e99,
    }
    data, code = post('/api/machines/CG-STONE-001/optimize', bad_body)
    all_ok &= check(code == 200, f"HTTP 200 (got {code})")
    all_ok &= check(data.get('degraded') is True,
                    f"degraded=True (got {data.get('degraded')}) "
                    f"— 应标记为降级因使用了非法客户端参数")
    all_ok &= check('warnings' in data and len(data['warnings']) > 0,
                    f"含 warnings 字段 (got {data.get('warnings')})")
    dir = data.get('adjustment_direction')
    all_ok &= check(dir in ['increase', 'decrease', 'maintain'],
                    f"direction 合法 (got {dir})")
    pur = data.get('predicted_purity_at_optimal')
    all_ok &= check(pur is not None and 90 <= pur <= 100 and isinstance(pur, float),
                    f"预测纯度合法 (got {pur})")

    section("4. POST /api/predict (非法参数)")
    bad_body2 = {
        'machine_id': 'CG-STONE-003',
        'parameters': {
            'vibration_freq': float('nan'),
            'inclination_angle': float('inf'),
            'fan_airflow': -1,
            'amplitude': {},
        },
    }
    data, code = post('/api/predict', bad_body2)
    all_ok &= check(code == 200, f"HTTP 200 (got {code})")
    pur = data.get('predicted_purity')
    all_ok &= check(pur is not None and 90 <= pur <= 100 and not (isinstance(pur, float) and (np.isnan(pur) or np.isinf(pur))),
                    f"predicted_purity 合法 (got {pur})")
    print(f"  返回 degraded={data.get('degraded')}, strategy={data.get('model_strategy')}")

    section("5. POST /api/data/refresh (注入 50% 脏数据)")
    data, code = post('/api/data/refresh', {'dirty_ratio': 0.5})
    all_ok &= check(code == 200, f"HTTP 200 (got {code})")
    all_ok &= check(data.get('status') == 'data_refreshed',
                    f"status=data_refreshed (got {data.get('status')})")

    section("6. GET /api/health (50% 脏数据后)")
    data, code = get('/api/health')
    all_ok &= check(code == 200, f"HTTP 200")
    # 注意：即便 50% 脏数据，只要清洗后仍能训练出模型，就不算 degraded
    status = data.get('status')
    all_ok &= check(status in ['healthy', 'degraded'],
                    f"status ∈ {{healthy, degraded}} (got {status})")
    dm = data.get('degraded_machines', -1)
    all_ok &= check(isinstance(dm, int) and dm >= 0,
                    f"degraded_machines 合法 (got {dm})")
    print(f"  状态: {status}, degraded_machines={dm}")

    section("7. GET /api/data-quality (50% 脏数据后)")
    data, code = get('/api/data-quality')
    all_ok &= check(code == 200, f"HTTP 200")
    reports = data.get('reports', {})
    for mid, rep in reports.items():
        qw = rep.get('quality_warning', '?')
        nr = rep.get('null_ratio', 0)
        print(f"  {mid}: null={nr:.1%}, warning={qw}")
        all_ok &= check(qw in ['OK', 'HIGH_NULL_RATIO', 'HIGH_OUTLIER_RATIO', 'POOR_QUALITY'],
                        f"{mid} 合法 quality_warning")

    section("8. GET /api/machines/<id>/purity-curve (50% 脏数据)")
    data, code = get('/api/machines/CG-STONE-002/purity-curve')
    all_ok &= check(code == 200, f"HTTP 200")
    all_ok &= check('model_strategy' in data, f"返回 model_strategy")
    strat = data.get('model_strategy')
    print(f"  使用策略: {strat}")
    n_amps = len(data.get('amplitudes', []))
    n_purs = len(data.get('predicted_purities', []))
    all_ok &= check(n_amps == n_purs == 50,
                    f"曲线各 50 点 (got {n_amps}/{n_purs})")
    cur_pur = data.get('current_purity')
    all_ok &= check(90 <= cur_pur <= 100, f"current_purity 合法 (got {cur_pur})")
    all_ok &= check(not any(p is None for p in data['predicted_purities']),
                    "无 None 预测")

    section("9. POST /api/predict (最极端脏输入)")
    worst_body = {
        'machine_id': 'CG-STONE-004',
        'parameters': {
            'vibration_freq': float('nan'),
            'inclination_angle': float('inf'),
            'amplitude': [],
            'fan_airflow': {'oops': {'even': 'worse'}},
        },
    }
    data, code = post('/api/predict', worst_body)
    all_ok &= check(code == 200, f"HTTP 200 (got {code})")
    pur = data.get('predicted_purity')
    all_ok &= check(pur is not None and 90 <= pur <= 100
                    and not (isinstance(pur, float) and (np.isnan(pur) or np.isinf(pur))),
                    f"predicted_purity 合法 (got {pur})")
    print(f"  degraded={data.get('degraded')}, strategy={data.get('model_strategy')}")

    section("10. 404 /api/not-exist (不应被误判为降级)")
    data, code = get('/api/not-exist')
    all_ok &= check(code == 404, f"HTTP 404 (got {code})")
    all_ok &= check(data.get('degraded') is False,
                    f"404 不应标记 degraded (got {data.get('degraded')})")

    section("11. GET /api/model/status")
    data, code = get('/api/model/status')
    all_ok &= check(code == 200, f"HTTP 200")
    for ms in data.get('machines', []):
        print(f"  {ms['machine_id']}: strategy={ms['model_strategy']}, fallback={ms.get('using_fallback')}")
        all_ok &= check(isinstance(ms.get('model_strategy'), str),
                        f"{ms['machine_id']} strategy 是字符串")

    print(f"\n{'=' * 60}")
    if all_ok:
        print("  🎉 全部端到端 API 测试通过！")
    else:
        print("  ❌ 存在测试失败，请检查上方标记。")
    print('=' * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
