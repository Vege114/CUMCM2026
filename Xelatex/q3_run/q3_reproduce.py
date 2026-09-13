"""第三问：退款计费、严格物理情景MILP、因果回放及独立核验。

运行：python q3_reproduce.py --case main（完整365日，其中1月预热）。
对照：python q3_reproduce.py --case no_update。
依赖：numpy scipy pandas matplotlib；所有输入在脚本同级inputs，无个人绝对路径。
冻结HGB预测是输入层已有成果；本程序重现第三问更新、调度与核验，不重新训练HGB。
"""
from __future__ import annotations
import argparse  # 解析命令行，不把本机路径写进程序。
import hashlib  # 锁定输入和程序的内容哈希。
import json  # 保存可追溯的参数、结果和审计。
import sys  # 输出明确失败状态。
import time  # 记录实际运行时间。
import warnings  # 只屏蔽HiGHS透传线程参数的已知提示。
from pathlib import Path  # 以程序目录定位随包输入。
from functools import lru_cache  # 复用已发布预报，不改变发布时间。
try:
    import numpy as np  # 数组、能量运算。
    import pandas as pd  # 读取附件并输出日期表。
    from scipy.optimize import milp, Bounds, LinearConstraint  # HiGHS混合整数求解器。
    from scipy.sparse import coo_matrix  # 大规模约束的稀疏表示。
except ImportError as exc:
    raise SystemExit('缺少依赖，请运行 python -m pip install numpy scipy pandas matplotlib') from exc

HERE = Path(__file__).resolve().parent  # 工作目录任意变化仍可定位输入。
DT, ETA, LOW, HIGH, LIMIT = 1/6, np.sqrt(.9), 1200., 10800., 5000/6
TOL = 2e-5  # kWh的数值核验容差，不放宽题面物理限制。


def save(path, obj):
    """输出JSON；把NumPy标量转为标准值，不把NaN写入结果。"""
    def cast(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=cast,
                              allow_nan=False), encoding='utf-8')


def read_csv(path):
    """兼容附件UTF-8/GB18030编码；读取失败时不伪造或填零数据。"""
    if not path.is_file(): raise FileNotFoundError(f'缺少随包输入：{path.name}')
    for encoding in ('utf-8-sig', 'gb18030'):
        try: return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError: continue
    raise ValueError(f'无法解码{path.name}，请恢复inputs中原始文件')


class Forecasts:
    """单HGB午夜输入、官方节点校准与同发布时间历史残差。"""
    def __init__(self, directory=HERE/'inputs'):
        self.directory = Path(directory)
        ref = read_csv(self.directory/'附件1.csv')
        self.reference = ref.iloc[:, 1:].to_numpy(float)
        frames = [read_csv(self.directory/name) for name in
                  ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')]
        expected = pd.date_range('2025-01-01', '2025-12-31')
        for frame in frames:
            if frame.shape != (365, 145): raise ValueError('年度附件应为365×145')
            if not pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(expected):
                raise ValueError('年度附件日期不连续或错位')
        self.actual = np.stack([f.iloc[:, 1:].to_numpy(float) for f in frames], -1)
        self.price = self.reference[:, 0].copy()
        if self.reference.shape != (144, 3): raise ValueError('附件1必须有144槽')
        if not np.isfinite(self.actual).all() or (self.actual < 0).any():
            raise ValueError('实际功率含缺失、负数或非有限值')
        if not np.isfinite(self.reference).all() or (self.price <= 0).any():
            raise ValueError('附件1含非法数据或非正电价')
        table = read_csv(self.directory/'附件3.csv')
        table.iloc[:, 0] = table.iloc[:, 0].ffill()
        self.official = {}
        for row in table.itertuples(index=False, name=None):
            day = (pd.Timestamp(row[0])-expected[0]).days
            slot = int(str(row[1]).split(':')[0])*6
            values = np.asarray(row[2:], float)
            if slot not in (0,36,72,108) or values.shape != (24,):
                raise ValueError('附件3发布时刻或节点数错误')
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('官方预报含非法值')
            self.official[day*144+slot] = values
        if len(self.official) != 1460: raise ValueError('官方发布缺失或重复')
        archive = self.directory/'direct_hgb_ridge28_memory_half.npz'
        try:
            with np.load(archive, allow_pickle=False) as z:
                self.midnight = z['values'].copy()
                if not np.array_equal(z['origins'], np.arange(31,365)*144):
                    raise ValueError('冻结预测发布时间不匹配')
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError('冻结预测损坏；请恢复随包NPZ，禁止以实况替代预测') from exc
        if self.midnight.shape != (334,144,2): raise ValueError('冻结预测形状错误')
        # 节点修正到十分钟平均功率的线性映射，锚点修正固定为零。
        self.basis = np.zeros((144,24))
        for h in range(24):
            knots = np.zeros(25); knots[h+1] = 1
            right = np.interp(np.arange(1,145), np.arange(0,145,6), knots)
            self.basis[:,h] = (np.r_[0.,right[:-1]]+right)/2
        difference = np.diff(np.eye(24), n=2, axis=0)
        self.penalty = 12*difference.T@difference+2*np.eye(24)

    def observed(self, begin, end, cutoff):
        """统一前缀读取器，当前及未来区间不作为已观察样本。"""
        if not 0 <= begin <= end <= cutoff: raise ValueError('越过信息截止时刻')
        return self.actual.reshape(-1,2)[begin:end].copy()

    @lru_cache(maxsize=366)
    def base(self, day):
        if day >= 31: return self.midnight[day-31].copy()
        value = self.reference[:,1:3].copy()  # 一月首日仅使用已给定附件1。
        for channel, lag in ((0,7),(1,1)):
            lag = 1 if lag == 7 and day < 7 else lag
            if day >= lag: value[:,channel] = self.actual[day-lag,:,channel]
        return value

    def integrate(self, hourly, origin):
        anchor = self.observed(origin-1,origin,origin)[0,1] if origin else 0.
        right = np.interp(np.arange(1,145),np.arange(0,145,6),np.r_[anchor,hourly])
        return (np.r_[anchor,right[:-1]]+right)/2

    @lru_cache(maxsize=1500)
    def get(self, day, slot):
        if slot not in (0,36,72,108): raise ValueError('仅允许四个发布时刻')
        origin = day*144+slot; n = 144-slot
        load = self.base(day)[slot:,0].copy()
        if slot:
            begin = max(0,slot-18)
            error = self.observed(day*144+begin,origin,origin)[:,0]-self.base(day)[begin:slot,0]
            weights = 2**(-np.arange(len(error)-1,-1,-1)/6)
            bias = np.average(error,weights=weights)*len(error)/(len(error)+12)
            load = np.maximum(0,load+bias*np.exp(-np.arange(n)/36))
        history = np.arange(max(origin%144,origin-28*144),origin,144)
        history = history[history+144 <= origin]  # 必须等完整24小时标签到达。
        correction = np.zeros(24)
        if len(history):
            errors = [self.observed(int(o),int(o)+144,origin)[:,1]
                      -self.integrate(self.official[int(o)],int(o)) for o in history]
            weights = 2**(-np.arange(len(history)-1,-1,-1)/14)
            mean = np.average(errors,axis=0,weights=weights)
            correction = np.linalg.solve(self.basis.T@self.basis+self.penalty,
                                         self.basis.T@mean)*len(history)/(len(history)+7)
        hourly = self.official[origin]
        corrected = np.where(hourly > 0,np.maximum(0,hourly+correction),0)
        return np.column_stack((load,self.integrate(corrected,origin)[:n]))

    def paths(self, day, slot, count=7):
        history = np.arange(max(1,day-28),day)
        if not len(history): return np.zeros((1,144-slot)), history
        history = history[np.linspace(0,len(history)-1,min(count,len(history))).astype(int)]
        errors = []
        for old in history:
            actual = self.observed(int(old)*144+slot,(int(old)+1)*144,day*144+slot)
            difference = actual-self.get(int(old),slot)
            errors.append((difference[:,0]-difference[:,1])*DT)
        return np.asarray(errors), history


class Model:
    """稀疏MILP装配；显式记录变量界、整数性及每行约束。"""
    def __init__(self):
        self.lo=[]; self.hi=[]; self.cost=[]; self.integer=[]
        self.rows=[]; self.cols=[]; self.vals=[]; self.lb=[]; self.ub=[]
    def var(self, shape, lo=0, hi=np.inf, cost=0, integer=0):
        start=len(self.lo); size=int(np.prod(shape))
        for target,value in ((self.lo,lo),(self.hi,hi),(self.cost,cost),(self.integer,integer)):
            target.extend(np.broadcast_to(value,shape).ravel())
        return np.arange(start,start+size).reshape(shape)
    def row(self, terms, lo=-np.inf, hi=np.inf):
        number=len(self.lb)
        for index,value in terms:
            self.rows.append(number); self.cols.append(int(index)); self.vals.append(value)
        self.lb.append(lo); self.ub.append(hi)
    def solve(self, seconds=2.):
        matrix=coo_matrix((self.vals,(self.rows,self.cols)),
                          shape=(len(self.lb),len(self.lo))).tocsc()
        matrix.indices=matrix.indices.astype(np.int32); matrix.indptr=matrix.indptr.astype(np.int32)
        start=time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            result=milp(np.asarray(self.cost),integrality=np.asarray(self.integer),
                        bounds=Bounds(self.lo,self.hi),
                        constraints=LinearConstraint(matrix,self.lb,self.ub),
                        options={'time_limit':seconds,'mip_rel_gap':.005,'threads':1})
        if result.x is None: raise RuntimeError('MILP没有可行解：'+result.message)
        x=result.x; ax=matrix@x; ints=np.asarray(self.integer,dtype=bool)
        residual=max(np.max(np.maximum(np.asarray(self.lo)-x,0)),
                     np.max(np.maximum(x-self.hi,0)),np.max(np.maximum(self.lb-ax,0)),
                     np.max(np.maximum(ax-self.ub,0)))
        int_error=np.max(np.abs(x[ints]-np.rint(x[ints]))) if ints.any() else 0.
        if residual > TOL or int_error > TOL: raise RuntimeError('MILP数值可行性检查失败')
        return x,{'status':int(result.status),'gap':float(result.mip_gap),
                  'seconds':time.perf_counter()-start,'residual':float(residual)}


def plan(prediction, price, initial, paths, original=None, scale=1., updates=True,
         final_day=False, seconds=2.):
    """退款情景MILP；场景补救为开环近似，实际动作只按当前实况执行。"""
    n=len(price); demand=(prediction[:,0]-prediction[:,1])*DT+scale*paths
    k=len(demand); model=Model()
    q=model.var((n,),cost=price if original is None else 0)
    if original is not None:
        up=model.var((n,),cost=1.5*price)
        down=model.var((n,),hi=original,cost=-.5*price)
        for t in range(n): model.row([(q[t],1),(up[t],-1),(down[t],1)],original[t],original[t])
    shortfall=np.full(n,5.); shortfall[36:]=1.5 if updates else 5.
    c=model.var((k,n),hi=LIMIT,cost=.005/k)
    d=model.var((k,n),hi=LIMIT,cost=.005/k)
    e=model.var((k,n),hi=np.maximum(demand,0),cost=shortfall*price/k)
    w=model.var((k,n)); state=model.var((k,n),lo=LOW,hi=HIGH)
    mode=model.var((k,n),hi=1,integer=1)
    variation=model.var((k,n),cost=.001/DT/k)
    terminal=0 if final_day else price.min()/ETA
    for index in state[:,-1]: model.cost[index]=-terminal/k
    for j in range(k):
        for t in range(n):
            model.row([(q[t],1),(d[j,t],1),(e[j,t],1),(c[j,t],-1),(w[j,t],-1)],demand[j,t],demand[j,t])
            terms=[(state[j,t],1),(c[j,t],-ETA),(d[j,t],1/ETA)]
            if t: terms.append((state[j,t-1],-1))
            rhs=initial if t==0 else 0.
            model.row(terms,rhs,rhs)
            model.row([(c[j,t],1),(mode[j,t],-LIMIT)],hi=0)
            model.row([(d[j,t],1),(mode[j,t],LIMIT)],hi=LIMIT)
            bound=max(demand[j,t],0.)
            model.row([(e[j,t],1),(mode[j,t],bound)],hi=bound)
            if t:
                diff=[(c[j,t],1),(d[j,t],-1),(c[j,t-1],-1),(d[j,t-1],1)]
                model.row(diff+[(variation[j,t],-1)],hi=0)
                model.row([(i,-v) for i,v in diff]+[(variation[j,t],-1)],hi=0)
    x,meta=model.solve(seconds)
    # 核验所有场景，不以平均充放电量掩盖单场景违约。
    meta['scenario_cd_overlap']=float(np.minimum(x[c],x[d]).max())
    meta['scenario_ce_overlap']=float(np.minimum(x[c],x[e]).max())
    if max(meta['scenario_cd_overlap'],meta['scenario_ce_overlap']) > TOL:
        raise RuntimeError('情景内发生禁止动作')
    return np.maximum(x[q],0),meta


def execute(q, actual, initial):
    """逐槽物理贪心反馈：只使用当槽实际功率，不读取后续实况。"""
    n=len(q); state=np.empty(n+1); state[0]=initial
    c,d,e,w=(np.zeros(n) for _ in range(4))
    for t in range(n):
        balance=q[t]+(actual[t,1]-actual[t,0])*DT
        if balance>=0:
            c[t]=min(balance,LIMIT,max(0,(HIGH-state[t])/ETA))
            if c[t]<20: c[t]=0.  # 沿用20kWh充电死区；不是最小硬件功率。
            w[t]=balance-c[t]
        else:
            d[t]=min(-balance,LIMIT,max(0,(state[t]-LOW)*ETA))
            e[t]=-balance-d[t]
        state[t+1]=state[t]+ETA*c[t]-d[t]/ETA
    return {'charge':c,'discharge':d,'emergency':e,'surplus':w,'states':state}


def fees(original, final, emergency, price):
    """原款、上调、退款、紧急四项；按最终净调整一次结算。"""
    return np.stack((original*price,1.5*price*np.maximum(final-original,0),
                     -.5*price*np.maximum(original-final,0),5*price*emergency),-1)


def verify(a):
    """独立按原始供需与状态方程重算，退款列允许为负。"""
    q,c,d,e,w,s=(a[key] for key in ('final','charge','discharge','emergency','surplus','states'))
    n=(a['actual'][...,0]-a['actual'][...,1])*DT
    balance=float(np.max(np.abs(q+d+e-c-w-n)))
    transition=float(np.max(np.abs(np.diff(s,axis=1)-ETA*c+d/ETA)))
    recalc=a['price']*(a['original']+1.5*np.maximum(q-a['original'],0)
                     -.5*np.maximum(a['original']-q,0)+5*e)
    billing=float(np.max(np.abs(recalc-a['fees'].sum(-1))))
    continuity=float(np.max(np.abs(s[1:,0]-s[:-1,-1]))) if len(s)>1 else 0.
    checks={'balance_kwh':balance,'state_kwh':transition,'billing_yuan':billing,
            'cross_day_kwh':continuity,'simultaneous_slots':int(((c>TOL)&(d>TOL)).sum()),
            'emergency_charging_slots':int(((c>TOL)&(e>TOL)).sum()),
            'min_soc':float(s.min()),'max_soc':float(s.max()),
            'max_charge_kw':float(c.max()/DT),'max_discharge_kw':float(d.max()/DT)}
    nonnegative=min(float(a[key].min()) for key in ('original','final','charge','discharge','emergency','surplus'))
    checks['passed']=bool(max(balance,transition,billing,continuity)<TOL
        and checks['simultaneous_slots']==checks['emergency_charging_slots']==0
        and s.min()>=LOW-TOL and s.max()<=HIGH+TOL and c.max()<=LIMIT+TOL
        and d.max()<=LIMIT+TOL and nonnegative>=-TOL)
    if not checks['passed']: raise RuntimeError('回放独立检查失败：'+str(checks))
    return checks


def day_run(forecasts, day, initial, updates=True, scale=1., seconds=2.):
    """日内只覆盖尚未执行部分；状态在相邻发布之间精确传递。"""
    starts=(0,36,72,108) if updates else (0,)
    out={k:np.zeros(144) for k in ('charge','discharge','emergency','surplus','final')}
    out['states']=np.empty(145); out['states'][0]=initial; logs=[]
    for start in starts:
        stop=min(144,start+36) if updates else 144
        prediction=forecasts.get(day,start)
        paths,history=forecasts.paths(day,start)
        purchase,meta=plan(prediction,forecasts.price[start:],out['states'][start],paths,
                           None if start==0 else out['original'][start:],scale,updates,day==364,seconds)
        if start==0: out['original']=purchase.copy()
        out['final'][start:]=purchase  # 已执行的[0,start)保持原值。
        segment=execute(purchase[:stop-start],forecasts.actual[day,start:stop],out['states'][start])
        for key in ('charge','discharge','emergency','surplus'): out[key][start:stop]=segment[key]
        out['states'][start:stop+1]=segment['states']
        logs.append({'day':day,'slot':start,'history_days':history.tolist(),**meta})
    out['price']=forecasts.price.copy(); out['actual']=forecasts.actual[day].copy()
    out['fees']=fees(out['original'],out['final'],out['emergency'],out['price'])
    return out,logs


def run(case='main', days=365, scale=1., seconds=2., directory=HERE/'inputs'):
    """完整自然年回放，前31日只预热；不沿用旧计费下的预热状态。"""
    out=HERE/'results'/case
    if (out/'summary.json').exists(): raise FileExistsError('结果已存在，请改case或另复制运行包')
    out.mkdir(parents=True,exist_ok=True)
    forecast=Forecasts(directory); initial=6000.; parts=[]; logs=[]; began=time.perf_counter()
    save(out/'protocol.json',{'case':case,'days':days,'scale':scale,'seconds':seconds,
        'updates':case!='no_update','refund':.5,'scenarios':7,'gap_target':.005,
        'evaluation':'February--December; chronological development evaluation',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(directory).iterdir() if p.is_file()}})
    for day in range(days):
        detail,audit=day_run(forecast,day,initial,case!='no_update',scale,seconds)
        initial=float(detail['states'][-1]); parts.append(detail); logs.extend(audit)
        if (day+1)%10==0 or day==days-1:
            print(case,day+1,'days',round(time.perf_counter()-began,1),'seconds',flush=True)
            save(out/'progress.json',{'days':day+1,'seconds':time.perf_counter()-began})
    arrays={key:np.stack([part[key] for part in parts]) for key in parts[0]}
    checks=verify(arrays); np.savez_compressed(out/'dispatch_all.npz',**arrays)
    formal={k:v[31:] for k,v in arrays.items()} if days>31 else arrays
    if days>31: verify(formal)
    net= (formal['charge']-formal['discharge']).ravel(); signs=np.sign(net[np.abs(net)>TOL])
    summary={'case':case,'evaluation_days':max(0,days-31),'complete':days==365,
        'total_cost':float(formal['fees'].sum()),'components':formal['fees'].sum((0,1)).tolist(),
        'emergency_kwh':float(formal['emergency'].sum()),
        'downward_kwh':float(np.maximum(formal['original']-formal['final'],0).sum()),
        'reversals':int((signs[1:]*signs[:-1]<0).sum()),
        'throughput_kwh':float((formal['charge']+formal['discharge']).sum()),
        'initial_soc':float(formal['states'][0,0]),'final_soc':float(formal['states'][-1,-1]),
        'wall_seconds':time.perf_counter()-began,'verification':checks,
        'max_gap':max(r['gap'] for r in logs),'mean_gap':float(np.mean([r['gap'] for r in logs])),
        'time_limit_solves':sum(r['status']==1 for r in logs),'solves':len(logs),
        'max_scenario_cd_overlap':max(r['scenario_cd_overlap'] for r in logs),
        'max_scenario_ce_overlap':max(r['scenario_ce_overlap'] for r in logs)}
    save(out/'solver_audit.json',logs); save(out/'summary.json',summary)
    dates=pd.date_range('2025-01-01',periods=days)
    table=pd.DataFrame({'date':dates,'cost_yuan':arrays['fees'].sum((1,2)),
                       'planned_yuan':arrays['fees'][:,:,0].sum(1),
                       'up_yuan':arrays['fees'][:,:,1].sum(1),
                       'refund_yuan':arrays['fees'][:,:,2].sum(1),
                       'emergency_yuan':arrays['fees'][:,:,3].sum(1)})
    table.to_csv(out/'daily.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='main')
    parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--inputs',type=Path,default=HERE/'inputs')
    args=parser.parse_args()
    try:
        if not 1<=args.days<=365 or args.scale<=0 or args.seconds<=0:
            raise ValueError('days须在1--365，scale和seconds须为正数')
        run(args.case,args.days,args.scale,args.seconds,args.inputs)
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'求解停止：{exc}。未生成成功标志；修复输入或参数后使用新的case重试。',file=sys.stderr)
        sys.exit(1)
