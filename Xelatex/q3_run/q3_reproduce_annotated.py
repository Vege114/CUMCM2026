"""第三问：退款计费、严格物理情景MILP、因果回放及独立核验。

运行：python q3_reproduce.py --case main（完整365日，其中1月预热）。
对照：python q3_reproduce.py --case no_update。
依赖：numpy scipy pandas matplotlib；所有输入在脚本同级inputs，无个人绝对路径。
冻结HGB预测是输入层已有成果；本程序重现第三问更新、调度与核验，不重新训练HGB。
"""
from __future__ import annotations  # 解析用户运行选项并保证失败时返回明确错误状态。
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
except ImportError as exc:  # 解析用户运行选项并保证失败时返回明确错误状态。
    raise SystemExit('缺少依赖，请运行 python -m pip install numpy scipy pandas matplotlib') from exc

HERE = Path(__file__).resolve().parent  # 工作目录任意变化仍可定位输入。
DT, ETA, LOW, HIGH, LIMIT = 1/6, np.sqrt(.9), 1200., 10800., 5000/6  # 解析用户运行选项并保证失败时返回明确错误状态。
TOL = 2e-5  # kWh的数值核验容差，不放宽题面物理限制。


def save(path, obj):  # 以UTF-8和标准JSON类型保存证据，禁止写入非有限数值。
    """输出JSON；把NumPy标量转为标准值，不把NaN写入结果。"""
    def cast(v):  # 将NumPy数组或标量转换成可移植的JSON对象。
        if isinstance(v, np.ndarray): return v.tolist()  # 条件判断：将NumPy数组或标量转换成可移植的JSON对象。
        if isinstance(v, np.generic): return v.item()  # 条件判断：将NumPy数组或标量转换成可移植的JSON对象。
        raise TypeError(type(v).__name__)  # 将NumPy数组或标量转换成可移植的JSON对象。
    path.parent.mkdir(parents=True, exist_ok=True)  # 以UTF-8和标准JSON类型保存证据，禁止写入非有限数值。
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=cast,  # 以UTF-8和标准JSON类型保存证据，禁止写入非有限数值。
                              allow_nan=False), encoding='utf-8')


def read_csv(path):  # 尝试允许的原始编码；读失败时显式报错而非虚构数据。
    """兼容附件UTF-8/GB18030编码；读取失败时不伪造或填零数据。"""
    if not path.is_file(): raise FileNotFoundError(f'缺少随包输入：{path.name}')  # 条件判断：尝试允许的原始编码；读失败时显式报错而非虚构数据。
    for encoding in ('utf-8-sig', 'gb18030'):
        try: return pd.read_csv(path, encoding=encoding)  # 尝试允许的原始编码；读失败时显式报错而非虚构数据。
        except UnicodeDecodeError: continue  # 尝试允许的原始编码；读失败时显式报错而非虚构数据。
    raise ValueError(f'无法解码{path.name}，请恢复inputs中原始文件')  # 尝试允许的原始编码；读失败时显式报错而非虚构数据。


class Forecasts:  # 组织发布时可用的负载、光伏和历史残差输入。
    """单HGB午夜输入、官方节点校准与同发布时间历史残差。"""
    def __init__(self, directory=HERE/'inputs'):
        self.directory = Path(directory)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        ref = read_csv(self.directory/'附件1.csv')
        self.reference = ref.iloc[:, 1:].to_numpy(float)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        frames = [read_csv(self.directory/name) for name in  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
                  ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')]
        expected = pd.date_range('2025-01-01', '2025-12-31')
        for frame in frames:  # 循环处理：初始化数据形状、时间轴与节点积分矩阵的严格检查。
            if frame.shape != (365, 145): raise ValueError('年度附件应为365×145')
            if not pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(expected):  # 条件判断：初始化数据形状、时间轴与节点积分矩阵的严格检查。
                raise ValueError('年度附件日期不连续或错位')
        self.actual = np.stack([f.iloc[:, 1:].to_numpy(float) for f in frames], -1)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        self.price = self.reference[:, 0].copy()  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        if self.reference.shape != (144, 3): raise ValueError('附件1必须有144槽')
        if not np.isfinite(self.actual).all() or (self.actual < 0).any():  # 条件判断：初始化数据形状、时间轴与节点积分矩阵的严格检查。
            raise ValueError('实际功率含缺失、负数或非有限值')
        if not np.isfinite(self.reference).all() or (self.price <= 0).any():  # 条件判断：初始化数据形状、时间轴与节点积分矩阵的严格检查。
            raise ValueError('附件1含非法数据或非正电价')
        table = read_csv(self.directory/'附件3.csv')
        table.iloc[:, 0] = table.iloc[:, 0].ffill()  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        self.official = {}  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        for row in table.itertuples(index=False, name=None):  # 循环处理：初始化数据形状、时间轴与节点积分矩阵的严格检查。
            day = (pd.Timestamp(row[0])-expected[0]).days  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
            slot = int(str(row[1]).split(':')[0])*6
            values = np.asarray(row[2:], float)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
            if slot not in (0,36,72,108) or values.shape != (24,):  # 条件判断：初始化数据形状、时间轴与节点积分矩阵的严格检查。
                raise ValueError('附件3发布时刻或节点数错误')
            if not np.isfinite(values).all() or (values < 0).any():  # 条件判断：初始化数据形状、时间轴与节点积分矩阵的严格检查。
                raise ValueError('官方预报含非法值')
            self.official[day*144+slot] = values  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        if len(self.official) != 1460: raise ValueError('官方发布缺失或重复')
        archive = self.directory/'direct_hgb_ridge28_memory_half.npz'
        try:
            with np.load(archive, allow_pickle=False) as z:  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
                self.midnight = z['values'].copy()
                if not np.array_equal(z['origins'], np.arange(31,365)*144):
                    raise ValueError('冻结预测发布时间不匹配')
        except (OSError, KeyError, ValueError) as exc:  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
            raise ValueError('冻结预测损坏；请恢复随包NPZ，禁止以实况替代预测') from exc
        if self.midnight.shape != (334,144,2): raise ValueError('冻结预测形状错误')
        # 节点修正到十分钟平均功率的线性映射，锚点修正固定为零。
        self.basis = np.zeros((144,24))  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        for h in range(24):  # 循环处理：初始化数据形状、时间轴与节点积分矩阵的严格检查。
            knots = np.zeros(25); knots[h+1] = 1  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
            right = np.interp(np.arange(1,145), np.arange(0,145,6), knots)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
            self.basis[:,h] = (np.r_[0.,right[:-1]]+right)/2  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        difference = np.diff(np.eye(24), n=2, axis=0)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        self.penalty = 12*difference.T@difference+2*np.eye(24)  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。

    def observed(self, begin, end, cutoff):  # 限定实测读取区间为截止时刻之前的前缀。
        """统一前缀读取器，当前及未来区间不作为已观察样本。"""
        if not 0 <= begin <= end <= cutoff: raise ValueError('越过信息截止时刻')
        return self.actual.reshape(-1,2)[begin:end].copy()  # 限定实测读取区间为截止时刻之前的前缀。

    @lru_cache(maxsize=366)  # 组织发布时可用的负载、光伏和历史残差输入。
    def base(self, day):  # 读取已冻结午夜预测；一月只使用已观察周期滞后。
        if day >= 31: return self.midnight[day-31].copy()  # 条件判断：读取已冻结午夜预测；一月只使用已观察周期滞后。
        value = self.reference[:,1:3].copy()  # 一月首日仅使用已给定附件1。
        for channel, lag in ((0,7),(1,1)):  # 循环处理：读取已冻结午夜预测；一月只使用已观察周期滞后。
            lag = 1 if lag == 7 and day < 7 else lag  # 读取已冻结午夜预测；一月只使用已观察周期滞后。
            if day >= lag: value[:,channel] = self.actual[day-lag,:,channel]  # 条件判断：读取已冻结午夜预测；一月只使用已观察周期滞后。
        return value  # 读取已冻结午夜预测；一月只使用已观察周期滞后。

    def integrate(self, hourly, origin):  # 用已观察锚点连接小时节点，按梯形积分形成十分钟平均功率。
        anchor = self.observed(origin-1,origin,origin)[0,1] if origin else 0.  # 用已观察锚点连接小时节点，按梯形积分形成十分钟平均功率。
        right = np.interp(np.arange(1,145),np.arange(0,145,6),np.r_[anchor,hourly])  # 用已观察锚点连接小时节点，按梯形积分形成十分钟平均功率。
        return (np.r_[anchor,right[:-1]]+right)/2  # 用已观察锚点连接小时节点，按梯形积分形成十分钟平均功率。

    @lru_cache(maxsize=1500)  # 组织发布时可用的负载、光伏和历史残差输入。
    def get(self, day, slot):  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        if slot not in (0,36,72,108): raise ValueError('仅允许四个发布时刻')
        origin = day*144+slot; n = 144-slot  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        load = self.base(day)[slot:,0].copy()  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        if slot:  # 条件判断：按当前发布时间执行负载前缀修正与光伏节点校准。
            begin = max(0,slot-18)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            error = self.observed(day*144+begin,origin,origin)[:,0]-self.base(day)[begin:slot,0]  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            weights = 2**(-np.arange(len(error)-1,-1,-1)/6)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            bias = np.average(error,weights=weights)*len(error)/(len(error)+12)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            load = np.maximum(0,load+bias*np.exp(-np.arange(n)/36))  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        history = np.arange(max(origin%144,origin-28*144),origin,144)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        history = history[history+144 <= origin]  # 必须等完整24小时标签到达。
        correction = np.zeros(24)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        if len(history):  # 条件判断：按当前发布时间执行负载前缀修正与光伏节点校准。
            errors = [self.observed(int(o),int(o)+144,origin)[:,1]  # 按当前发布时间执行负载前缀修正与光伏节点校准。
                      -self.integrate(self.official[int(o)],int(o)) for o in history]  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            weights = 2**(-np.arange(len(history)-1,-1,-1)/14)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            mean = np.average(errors,axis=0,weights=weights)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
            correction = np.linalg.solve(self.basis.T@self.basis+self.penalty,  # 按当前发布时间执行负载前缀修正与光伏节点校准。
                                         self.basis.T@mean)*len(history)/(len(history)+7)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        hourly = self.official[origin]  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        corrected = np.where(hourly > 0,np.maximum(0,hourly+correction),0)  # 按当前发布时间执行负载前缀修正与光伏节点校准。
        return np.column_stack((load,self.integrate(corrected,origin)[:n]))  # 按当前发布时间执行负载前缀修正与光伏节点校准。

    def paths(self, day, slot, count=7):  # 从已完成日提取相同发布时间的整条净需求误差路径。
        history = np.arange(max(1,day-28),day)  # 从已完成日提取相同发布时间的整条净需求误差路径。
        if not len(history): return np.zeros((1,144-slot)), history  # 条件判断：从已完成日提取相同发布时间的整条净需求误差路径。
        history = history[np.linspace(0,len(history)-1,min(count,len(history))).astype(int)]  # 从已完成日提取相同发布时间的整条净需求误差路径。
        errors = []  # 从已完成日提取相同发布时间的整条净需求误差路径。
        for old in history:  # 循环处理：从已完成日提取相同发布时间的整条净需求误差路径。
            actual = self.observed(int(old)*144+slot,(int(old)+1)*144,day*144+slot)  # 从已完成日提取相同发布时间的整条净需求误差路径。
            difference = actual-self.get(int(old),slot)  # 从已完成日提取相同发布时间的整条净需求误差路径。
            errors.append((difference[:,0]-difference[:,1])*DT)  # 从已完成日提取相同发布时间的整条净需求误差路径。
        return np.asarray(errors), history  # 从已完成日提取相同发布时间的整条净需求误差路径。


class Model:  # 解析用户运行选项并保证失败时返回明确错误状态。
    """稀疏MILP装配；显式记录变量界、整数性及每行约束。"""
    def __init__(self):  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        self.lo=[]; self.hi=[]; self.cost=[]; self.integer=[]  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
        self.rows=[]; self.cols=[]; self.vals=[]; self.lb=[]; self.ub=[]  # 初始化数据形状、时间轴与节点积分矩阵的严格检查。
    def var(self, shape, lo=0, hi=np.inf, cost=0, integer=0):  # 批量设置变量的上下界、目标系数与整数标识。
        start=len(self.lo); size=int(np.prod(shape))  # 批量设置变量的上下界、目标系数与整数标识。
        for target,value in ((self.lo,lo),(self.hi,hi),(self.cost,cost),(self.integer,integer)):  # 循环处理：批量设置变量的上下界、目标系数与整数标识。
            target.extend(np.broadcast_to(value,shape).ravel())  # 批量设置变量的上下界、目标系数与整数标识。
        return np.arange(start,start+size).reshape(shape)  # 批量设置变量的上下界、目标系数与整数标识。
    def row(self, terms, lo=-np.inf, hi=np.inf):  # 把一条线性约束写入稀疏矩阵及相应双侧界。
        number=len(self.lb)  # 把一条线性约束写入稀疏矩阵及相应双侧界。
        for index,value in terms:  # 循环处理：把一条线性约束写入稀疏矩阵及相应双侧界。
            self.rows.append(number); self.cols.append(int(index)); self.vals.append(value)  # 把一条线性约束写入稀疏矩阵及相应双侧界。
        self.lb.append(lo); self.ub.append(hi)  # 把一条线性约束写入稀疏矩阵及相应双侧界。
    def solve(self, seconds=2.):  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        matrix=coo_matrix((self.vals,(self.rows,self.cols)),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                          shape=(len(self.lb),len(self.lo))).tocsc()  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        matrix.indices=matrix.indices.astype(np.int32); matrix.indptr=matrix.indptr.astype(np.int32)  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        start=time.perf_counter()  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        with warnings.catch_warnings():  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            result=milp(np.asarray(self.cost),integrality=np.asarray(self.integer),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                        bounds=Bounds(self.lo,self.hi),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                        constraints=LinearConstraint(matrix,self.lb,self.ub),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                        options={'time_limit':seconds,'mip_rel_gap':.005,'threads':1})
        if result.x is None: raise RuntimeError('MILP没有可行解：'+result.message)
        x=result.x; ax=matrix@x; ints=np.asarray(self.integer,dtype=bool)  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        residual=max(np.max(np.maximum(np.asarray(self.lo)-x,0)),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                     np.max(np.maximum(x-self.hi,0)),np.max(np.maximum(self.lb-ax,0)),  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
                     np.max(np.maximum(ax-self.ub,0)))  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        int_error=np.max(np.abs(x[ints]-np.rint(x[ints]))) if ints.any() else 0.  # 调用HiGHS并独立核验所得解的界、整数性及线性残差。
        if residual > TOL or int_error > TOL: raise RuntimeError('MILP数值可行性检查失败')
        return x,{'status':int(result.status),'gap':float(result.mip_gap),
                  'seconds':time.perf_counter()-start,'residual':float(residual)}


def plan(prediction, price, initial, paths, original=None, scale=1., updates=True,  # 装配退款计费及每条场景均严格物理可行的MILP。
         final_day=False, seconds=2.):  # 装配退款计费及每条场景均严格物理可行的MILP。
    """退款情景MILP；场景补救为开环近似，实际动作只按当前实况执行。"""
    n=len(price); demand=(prediction[:,0]-prediction[:,1])*DT+scale*paths  # 装配退款计费及每条场景均严格物理可行的MILP。
    k=len(demand); model=Model()  # 装配退款计费及每条场景均严格物理可行的MILP。
    q=model.var((n,),cost=price if original is None else 0)  # 装配退款计费及每条场景均严格物理可行的MILP。
    if original is not None:  # 条件判断：装配退款计费及每条场景均严格物理可行的MILP。
        up=model.var((n,),cost=1.5*price)  # 新增常规购电每kWh按基础电价的1.5倍计价。
        down=model.var((n,),hi=original,cost=-.5*price)  # 下调系数为负：每取消1kWh退回对应原费用的50%。
        for t in range(n): model.row([(q[t],1),(up[t],-1),(down[t],1)],original[t],original[t])  # 循环处理：装配退款计费及每条场景均严格物理可行的MILP。
    shortfall=np.full(n,5.); shortfall[36:]=1.5 if updates else 5.  # 下次合法更新之后采用1.5倍机会代理；无更新对照保持5倍。
    c=model.var((k,n),hi=LIMIT,cost=.005/k)  # 装配退款计费及每条场景均严格物理可行的MILP。
    d=model.var((k,n),hi=LIMIT,cost=.005/k)  # 装配退款计费及每条场景均严格物理可行的MILP。
    e=model.var((k,n),hi=np.maximum(demand,0),cost=shortfall*price/k)  # 装配退款计费及每条场景均严格物理可行的MILP。
    w=model.var((k,n)); state=model.var((k,n),lo=LOW,hi=HIGH)  # 装配退款计费及每条场景均严格物理可行的MILP。
    mode=model.var((k,n),hi=1,integer=1)  # 装配退款计费及每条场景均严格物理可行的MILP。
    variation=model.var((k,n),cost=.001/DT/k)  # 装配退款计费及每条场景均严格物理可行的MILP。
    terminal=0 if final_day else price.min()/ETA  # 普通日赋予末库存机会价值，正式末日不再抵扣终值。
    for index in state[:,-1]: model.cost[index]=-terminal/k  # 循环处理：装配退款计费及每条场景均严格物理可行的MILP。
    for j in range(k):  # 循环处理：装配退款计费及每条场景均严格物理可行的MILP。
        for t in range(n):  # 循环处理：装配退款计费及每条场景均严格物理可行的MILP。
            model.row([(q[t],1),(d[j,t],1),(e[j,t],1),(c[j,t],-1),(w[j,t],-1)],demand[j,t],demand[j,t])  # 装配退款计费及每条场景均严格物理可行的MILP。
            terms=[(state[j,t],1),(c[j,t],-ETA),(d[j,t],1/ETA)]  # 装配退款计费及每条场景均严格物理可行的MILP。
            if t: terms.append((state[j,t-1],-1))  # 条件判断：装配退款计费及每条场景均严格物理可行的MILP。
            rhs=initial if t==0 else 0.  # 装配退款计费及每条场景均严格物理可行的MILP。
            model.row(terms,rhs,rhs)  # 装配退款计费及每条场景均严格物理可行的MILP。
            model.row([(c[j,t],1),(mode[j,t],-LIMIT)],hi=0)  # 使用二元模式将单槽充电限制在额定功率对应电量内。
            model.row([(d[j,t],1),(mode[j,t],LIMIT)],hi=LIMIT)  # 放电只允许在与充电互斥的另一模式内发生。
            bound=max(demand[j,t],0.)  # 装配退款计费及每条场景均严格物理可行的MILP。
            model.row([(e[j,t],1),(mode[j,t],bound)],hi=bound)  # 约束紧急购电仅能出现在非充电模式，上界取情景净需求正部。
            if t:  # 条件判断：装配退款计费及每条场景均严格物理可行的MILP。
                diff=[(c[j,t],1),(d[j,t],-1),(c[j,t-1],-1),(d[j,t-1],1)]  # 装配退款计费及每条场景均严格物理可行的MILP。
                model.row(diff+[(variation[j,t],-1)],hi=0)  # 装配退款计费及每条场景均严格物理可行的MILP。
                model.row([(i,-v) for i,v in diff]+[(variation[j,t],-1)],hi=0)  # 装配退款计费及每条场景均严格物理可行的MILP。
    x,meta=model.solve(seconds)  # 装配退款计费及每条场景均严格物理可行的MILP。
    # 核验所有场景，不以平均充放电量掩盖单场景违约。
    meta['scenario_cd_overlap']=float(np.minimum(x[c],x[d]).max())
    meta['scenario_ce_overlap']=float(np.minimum(x[c],x[e]).max())
    if max(meta['scenario_cd_overlap'],meta['scenario_ce_overlap']) > TOL:
        raise RuntimeError('情景内发生禁止动作')
    return np.maximum(x[q],0),meta  # 装配退款计费及每条场景均严格物理可行的MILP。


def execute(q, actual, initial):  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
    """逐槽物理贪心反馈：只使用当槽实际功率，不读取后续实况。"""
    n=len(q); state=np.empty(n+1); state[0]=initial  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
    c,d,e,w=(np.zeros(n) for _ in range(4))  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
    for t in range(n):  # 循环处理：按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
        balance=q[t]+(actual[t,1]-actual[t,0])*DT  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
        if balance>=0:  # 条件判断：按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
            c[t]=min(balance,LIMIT,max(0,(HIGH-state[t])/ETA))  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
            if c[t]<20: c[t]=0.  # 沿用20kWh充电死区；不是最小硬件功率。
            w[t]=balance-c[t]  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
        else:
            d[t]=min(-balance,LIMIT,max(0,(state[t]-LOW)*ETA))  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
            e[t]=-balance-d[t]  # 按当槽实际余缺裁剪储能动作，保持供需和SOC一致。
        state[t+1]=state[t]+ETA*c[t]-d[t]/ETA  # 按充电乘效率、放电除效率更新电池内部储电量。
    return {'charge':c,'discharge':d,'emergency':e,'surplus':w,'states':state}


def fees(original, final, emergency, price):  # 分别计算原款、上调追加费、下调退款与紧急购电费。
    """原款、上调、退款、紧急四项；按最终净调整一次结算。"""
    return np.stack((original*price,1.5*price*np.maximum(final-original,0),  # 分别计算原款、上调追加费、下调退款与紧急购电费。
                     -.5*price*np.maximum(original-final,0),5*price*emergency),-1)  # 分别计算原款、上调追加费、下调退款与紧急购电费。


def verify(a):  # 从实际轨迹重新计算方程残差及禁止动作数量。
    """独立按原始供需与状态方程重算，退款列允许为负。"""
    q,c,d,e,w,s=(a[key] for key in ('final','charge','discharge','emergency','surplus','states'))
    n=(a['actual'][...,0]-a['actual'][...,1])*DT
    balance=float(np.max(np.abs(q+d+e-c-w-n)))  # 从实际轨迹重新计算方程残差及禁止动作数量。
    transition=float(np.max(np.abs(np.diff(s,axis=1)-ETA*c+d/ETA)))  # 从实际轨迹重新计算方程残差及禁止动作数量。
    recalc=a['price']*(a['original']+1.5*np.maximum(q-a['original'],0)
                     -.5*np.maximum(a['original']-q,0)+5*e)
    billing=float(np.max(np.abs(recalc-a['fees'].sum(-1))))
    continuity=float(np.max(np.abs(s[1:,0]-s[:-1,-1]))) if len(s)>1 else 0.  # 从实际轨迹重新计算方程残差及禁止动作数量。
    checks={'balance_kwh':balance,'state_kwh':transition,'billing_yuan':billing,
            'cross_day_kwh':continuity,'simultaneous_slots':int(((c>TOL)&(d>TOL)).sum()),
            'emergency_charging_slots':int(((c>TOL)&(e>TOL)).sum()),
            'min_soc':float(s.min()),'max_soc':float(s.max()),
            'max_charge_kw':float(c.max()/DT),'max_discharge_kw':float(d.max()/DT)}
    nonnegative=min(float(a[key].min()) for key in ('original','final','charge','discharge','emergency','surplus'))
    checks['passed']=bool(max(balance,transition,billing,continuity)<TOL
        and checks['simultaneous_slots']==checks['emergency_charging_slots']==0
        and s.min()>=LOW-TOL and s.max()<=HIGH+TOL and c.max()<=LIMIT+TOL  # 从实际轨迹重新计算方程残差及禁止动作数量。
        and d.max()<=LIMIT+TOL and nonnegative>=-TOL)  # 从实际轨迹重新计算方程残差及禁止动作数量。
    if not checks['passed']: raise RuntimeError('回放独立检查失败：'+str(checks))
    return checks  # 从实际轨迹重新计算方程残差及禁止动作数量。


def day_run(forecasts, day, initial, updates=True, scale=1., seconds=2.):  # 只重规划未执行区间，将实际SOC传给下一合法发布。
    """日内只覆盖尚未执行部分；状态在相邻发布之间精确传递。"""
    starts=(0,36,72,108) if updates else (0,)  # 只重规划未执行区间，将实际SOC传给下一合法发布。
    out={k:np.zeros(144) for k in ('charge','discharge','emergency','surplus','final')}
    out['states']=np.empty(145); out['states'][0]=initial; logs=[]
    for start in starts:  # 循环处理：只重规划未执行区间，将实际SOC传给下一合法发布。
        stop=min(144,start+36) if updates else 144  # 只重规划未执行区间，将实际SOC传给下一合法发布。
        prediction=forecasts.get(day,start)  # 只重规划未执行区间，将实际SOC传给下一合法发布。
        paths,history=forecasts.paths(day,start)  # 只重规划未执行区间，将实际SOC传给下一合法发布。
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
    return out,logs  # 只重规划未执行区间，将实际SOC传给下一合法发布。


def run(case='main', days=365, scale=1., seconds=2., directory=HERE/'inputs'):
    """完整自然年回放，前31日只预热；不沿用旧计费下的预热状态。"""
    out=HERE/'results'/case
    if (out/'summary.json').exists(): raise FileExistsError('结果已存在，请改case或另复制运行包')
    out.mkdir(parents=True,exist_ok=True)  # 连续运行一月预热与334日正式评价，并归档完整证据。
    forecast=Forecasts(directory); initial=6000.; parts=[]; logs=[]; began=time.perf_counter()  # 连续运行一月预热与334日正式评价，并归档完整证据。
    save(out/'protocol.json',{'case':case,'days':days,'scale':scale,'seconds':seconds,
        'updates':case!='no_update','refund':.5,'scenarios':7,'gap_target':.005,
        'evaluation':'February--December; chronological development evaluation',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(directory).iterdir() if p.is_file()}})
    for day in range(days):  # 循环处理：连续运行一月预热与334日正式评价，并归档完整证据。
        detail,audit=day_run(forecast,day,initial,case!='no_update',scale,seconds)
        initial=float(detail['states'][-1]); parts.append(detail); logs.extend(audit)
        if (day+1)%10==0 or day==days-1:  # 条件判断：连续运行一月预热与334日正式评价，并归档完整证据。
            print(case,day+1,'days',round(time.perf_counter()-began,1),'seconds',flush=True)
            save(out/'progress.json',{'days':day+1,'seconds':time.perf_counter()-began})
    arrays={key:np.stack([part[key] for part in parts]) for key in parts[0]}  # 连续运行一月预热与334日正式评价，并归档完整证据。
    checks=verify(arrays); np.savez_compressed(out/'dispatch_all.npz',**arrays)
    formal={k:v[31:] for k,v in arrays.items()} if days>31 else arrays  # 连续运行一月预热与334日正式评价，并归档完整证据。
    if days>31: verify(formal)  # 对不含一月的正式评价区间再次检查物理与计费一致性。
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
    print(json.dumps(summary,ensure_ascii=False),flush=True)  # 连续运行一月预热与334日正式评价，并归档完整证据。
    return summary  # 连续运行一月预热与334日正式评价，并归档完整证据。


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)  # 解析用户运行选项并保证失败时返回明确错误状态。
    parser.add_argument('--case',default='main')
    parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--inputs',type=Path,default=HERE/'inputs')
    args=parser.parse_args()  # 解析用户运行选项并保证失败时返回明确错误状态。
    try:
        if not 1<=args.days<=365 or args.scale<=0 or args.seconds<=0:  # 条件判断：解析用户运行选项并保证失败时返回明确错误状态。
            raise ValueError('days须在1--365，scale和seconds须为正数')
        run(args.case,args.days,args.scale,args.seconds,args.inputs)  # 解析用户运行选项并保证失败时返回明确错误状态。
    except (OSError,ValueError,RuntimeError) as exc:  # 解析用户运行选项并保证失败时返回明确错误状态。
        print(f'求解停止：{exc}。未生成成功标志；修复输入或参数后使用新的case重试。',file=sys.stderr)  # 解析用户运行选项并保证失败时返回明确错误状态。
        sys.exit(1)  # 解析用户运行选项并保证失败时返回明确错误状态。
