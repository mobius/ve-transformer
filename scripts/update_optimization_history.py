"""Publish cumulative measured milestones and an exhaustive iteration index.

The full index quotes existing public reports; it does not turn pending work
or historical forecasts into measured performance. Curated milestones retain
measurement scope and rejected candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Every number is transcribed from the linked accepted report. Different rows
# have different workloads; reductions must not be added or multiplied.
MILESTONES = [
 ('20261002T062015Z-optimization-progress.md','最小Transformer：NLC线性','相对基础VE，三档约3.2–3.6倍','三卡独立float64对照通过，线性计算先获得收益。'),
 ('20261002T064702Z-attention-progress.md','最小Transformer：注意力NLC','128×256约14.457→0.921ms；三档约3.3/15.8/22.2倍','注意力矩阵化有效，完整三卡验证通过。'),
 ('20261002T071128Z-workspace-progress.md','最小Transformer：工作区','槽1三档延迟降低10.37/22.04/28.28%','主要收益来自移除整体清零，不能全部归因于缓冲复用。'),
 ('20261002T072334Z-trained-lm-progress.md','TinyStories训练权重基线','VE生成0.55–0.60词元/s','真实模型正确性通过，但旧逐次启动桥接没有端到端优势。'),
 ('20261002T075034Z-resident-progress.md','TinyStories常驻worker','0.5758→309–318词元/s','常驻消除了反复启动代价；此路径嵌入和输出头仍在主机。'),
 ('20261002T132803Z-qwen36-route-measured.md','Qwen3.6直接量化候选（后来拒绝）','短测约1.464词元/s；后续完整词表套件退出1','短测速度较高，但后续精度失败，不能采用为合格基线。'),
 ('20261002T154644Z-qwen36-dense-cache.md','Qwen3.6首次稠密缓存','单提示0.103386→0.506802词元/s；首次输入19.743→100.621s','热生成获益但冷填充代价明显，后续需要并行填充。'),
 ('20261002T175953Z-qwen36-final-acceptance.md','Qwen3.6：稠密缓存/并行填充','同精度生成0.1035→0.5096词元/s；首次输入约99→45s','固定35B-A3B单VE文本负载通过，缓存改善带宽与冷态填充。'),
 ('20261006T033737Z-qwen15-results.md','Qwen2.5-1.5B量化基线','最佳实测3.7762词元/s；24步CPU分数检查','高精度路径通过，原量化路径精度不合格不能采用其速度。'),
 ('20261006T045223Z-qwen15-session-results.md','Qwen1.5B常驻请求','热请求49.627→10.883s；热生成3.840词元/s','请求复用降低启动/加载成本，不等同于生成算子加速。'),
 ('20261006T052515Z-qwen15-attention-nlc.md','Qwen1.5B全注意力候选','输入约2.55→1.23s；生成约3.79→3.60词元/s','输入改善但生成回退，改为只优化多列预填充。'),
 ('20261006T055403Z-qwen15-compute-optimization-results.md','Qwen1.5B选择式注意力+1024行投影','生成3.810→4.502词元/s（+18.16%）；请求10.955→8.406s','只加速预填充注意力并扩大投影块，保持原精度。'),
 ('20261006T095108Z-qwen15-tile-sweep.md','Qwen1.5B分块/线程筛选','512/2048行生成−1.48/−0.58%；2/8线程−25.55/−22.91%','四个候选未胜过1024行、4线程，全部拒绝。'),
 ('20261006T103723Z-qwen15-diagnostic.md','Qwen1.5B诊断开销','首次逐次计时约4.495→3.311词元/s（−26.35%）','全量计时扰动过大，改为稀疏采样，失败测量不作瓶颈依据。'),
 ('20261006T110253Z-qwen15-input-reuse.md','Qwen1.5B输入整理复用','生成4.5047→4.8963词元/s（+8.69%）；请求8.395→7.537s','保持精度并复用算子内输入转换，正式对照通过。'),
 ('20261007T011801Z-qwen36-mtp-resume.md','Qwen3.6原生MTP','代码提示0.58481→0.93227词元/s（+59.42%）；中文短测−10.8/−13.7%','MTP收益依赖提示；代码对照有效，中文保留普通生成。'),
 ('20261008T020107Z-qwen36-mtp-session.md','MTP常驻/自动回退','12请求功能通过；代码普通/固定/自动0.593/0.880/0.783词元/s（单样本）','会话与回退功能通过，自动策略未证明额外收益且不默认采用。'),
 ('20261008T091957Z-sd-turbo-final-validation.md','SD-Turbo完整模型验证','单VE、FP32、512×512；六提示/四步CPU参考通过','完整图像链路可用，随后开始分项优化。'),
 ('20261008T102317Z-sd-turbo-pool-verified.md','SD-Turbo持久线程池','图191.826→139.215s（−27.43%）；进程264.240→212.289s','复用图计算线程降低通用阶段开销，含加载的进程指标另列。'),
 ('20261008T111611Z-sd-turbo-silu.md','SD-Turbo首个SiLU候选（拒绝）','极端输入−1000错误返回−1000；VAE出现非有限值','未经范围检查的向量数学不合格，失败耗时不作为收益。'),
 ('20261008T114450Z-sd-turbo-silu-verified.md','SD-Turbo受检查SiLU','图145.630→96.387s（−33.81%）；进程216.835→167.596s','修正范围/除法编译选项后通过完整验证。'),
 ('20261009T031452Z-sd-turbo-softmax-verified.md','SD-Turbo Softmax','图96.638→51.655s（−46.55%）；进程169.065→125.915s','仅向量化exp并保持累加顺序，模型参考通过。'),
 ('20261009T053046Z-sd-turbo-im2col-verified.md','SD-Turbo im2col','图59.277→20.512s（−65.40%）；进程131.625→92.942s','卷积输入整理循环优化有效，数据布局和数学结果保持。'),
 ('20261009T053636Z-sd-turbo-resident-report.md','SD-Turbo初版权重常驻（未采用）','多组件独立池下请求长期未完成，未形成有效速度','停顿候选不作为收益，后续改为共享池。'),
 ('20261009T071320Z-sd-turbo-resident-verified.md','SD-Turbo共享池+权重常驻','热请求92.701→32.042s（−65.43%）；两请求进程185.970→124.917s','保留权重减少后续加载，增加节点内存占用。'),
 ('20261009T080313Z-sd-turbo-tokenizer-verified.md','SD-Turbo分词器复用','热请求32.539→22.407s（−31.14%）','复用初始化词表，后续编码仍执行并核验输出。'),
 ('20261009T084121Z-nlc-gemm-verified.md','NLC矩阵独立线程筛选','40个矩阵/线程配置；VAE代表形状4线程更快','仅为独立矩阵证据，不能外推整模型加速。'),
 ('20261009T085621Z-sd-turbo-nlc-staged.md','混合pthread/NLC分阶段线程（拒绝）','重试180s仅完成3个UNet图；准备82.36s','独立运行时切换导致严重停顿，未采用。'),
 ('20261009T094430Z-sd-turbo-nlc-abba.md','统一OpenMP 8线程','热请求22.384→18.194s（−18.72%）','统一运行时解除停顿，双构建固定源对照通过。'),
 ('20261009T104156Z-sd-turbo-binary-abba.md','空间平面广播加乘','热请求18.157→13.943s（−23.21%）','长连续循环减少广播开销，输出字节一致。'),
 ('20261009T105426Z-ve-omp-switch.md','统一线程切换探针','8→4约0.352ms，4→8约8.142ms','恢复8线程存在成本，必须在整图中验证取舍。'),
 ('20261009T111710Z-sd-turbo-vae-abba.md','全VAE 4线程','热请求13.976→13.059s（−6.56%）','矩阵获益超过通用算子损失，显式配置通过。'),
 ('20261009T113726Z-sd-turbo-vae-selective-abba.md','通用8/VAE矩阵4','热请求13.081→12.292s（−6.03%）','恢复通用8线程的收益超过每图切换成本。'),
 ('20261009T114825Z-sd-turbo-selective-op-profile.md','新配置瓶颈诊断','CLIP/UNet GELU0.761/0.733s；VAE im2col0.810s','GELU成为下一目标，诊断耗时不混入正式收益。'),
 ('20261009T115234Z-sd-turbo-gelu.md','GELU独立向量核','102用例通过，最大误差4.45e-7；此前两次退出132','标量参考测试下相同向量核通过，失败根因尚未确定。'),
]


def cell(text):
    return re.sub(r'\s+',' ',text).replace('|','\\|').strip()


def source_excerpt(text):
    # Preserve actual words, including failure/pending qualifiers. No automatic
    # conversion of a forecast or incomplete check into a measured result.
    paragraphs = [re.sub(r'\s+',' ',p).strip() for p in text.split('\n\n') if not p.lstrip().startswith('#')]
    candidates = [p for p in paragraphs if re.search(r'\d.*(?:ms|秒|词元|token/s|%|误差|通过|失败|PASS)',p)]
    if not candidates:
        return '无独立性能数字；该轮记录实现、环境或待验证事项。'
    def score(p):
        return sum(p.count(word)*weight for word,weight in [('→',9),('均值',6),('中位',6),('吞吐',5),('token/s',5),('词元/秒',5),('通过',2),('失败',2)])
    selected = max(candidates,key=score)
    selected = re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',selected)
    return selected[:230]+('…（原文摘录）' if len(selected)>230 else '（原文摘录）')


def source_summary(text, title):
    sentences = [re.sub(r'\s+',' ',part).strip() for part in re.split(r'。|\n\n',text)]
    candidates = [part for part in sentences if not part.startswith('#') and any(word in part for word in ('最终','通过','失败','未完成','未采用','尚未','修正','验证'))]
    if not candidates:
        return '该轮记录“'+title+'”的实施状态，未新增可独立使用的性能结论。'
    def score(part):
        return sum(part.count(word)*weight for word,weight in [('最终',7),('通过',4),('未采用',6),('失败',4),('实测',3),('尚未',2),('未完成',2)])
    selected=max(candidates,key=score)
    selected=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',selected)
    return selected[:170]+('…（原文结论摘录）' if len(selected)>170 else '。（原文结论摘录）')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--current-report',type=Path,required=True)
    parser.add_argument('--next-direction',required=True)
    args=parser.parse_args()
    current=args.current_report.resolve();current.relative_to(ROOT/'docs/impl')
    stamp=current.name.split('-')[0]
    groups={}
    for p in sorted((ROOT/'docs/impl').glob('*.md')):
        if p.name.endswith('-optimization-history.md'):continue
        groups.setdefault(p.name[:16],[]).append(p)
    milestones=[]
    for name,title,data,summary in MILESTONES:
        p=ROOT/'docs/impl'/name
        if not p.exists():raise RuntimeError('missing milestone source: '+name)
        milestones.append({'report':'docs/impl/'+name,'title':title,'data':data,'summary':summary,'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    gelu=ROOT/'docs/results/20261009T115851Z-sd-turbo-gelu-model.json'
    if gelu.exists():
        v=json.loads(gelu.read_text());rows=v.get('tests',[])
        detail=str(v['cpu_stage_checks'])+'项模型检查、102项对象检查'
        for row in rows:
            if row['name']=='six':
                hot=row['request_seconds'][1:];detail+='；一步热%.3f–%.3fs'%(min(hot),max(hot))
            if row['name']=='four':detail+='；四步热%.3fs'%row['request_seconds'][1]
        gelu_formal=sorted((ROOT/'docs/results').glob('*-sd-turbo-gelu-abba.json'))
        detail+=('；正确性轮未单独测收益，正式对照见后续轮' if gelu_formal else '；正式收益待交替对照')
        name='20261009T115851Z-sd-turbo-gelu-model.md';p=ROOT/'docs/impl'/name
        milestones.append({'report':'docs/impl/'+name,'title':('当前：GELU模型接入' if current.name==name else 'GELU模型接入（正确性轮）'),'data':detail,'summary':('多样正确性通过，正式性能结果另列。' if gelu_formal else '多样正确性通过，原/向量GELU同二进制对照正在运行。'),'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gelu-abba.json')):
        v=json.loads(proof.read_text());metric=v['metrics']['request1'];name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        data='一步热%.3f→%.3fs（延迟降低%.2f%%）；%d项CPU检查'%(metric['generic_seconds'],metric['ve_seconds'],metric['latency_reduction_percent'],v['total_cpu_stage_checks'])
        milestones.append({'report':'docs/impl/'+name,'title':('当前：GELU正式对照' if current.name==name else 'GELU正式对照'),'data':data,
                           'summary':('同二进制对照确认收益，仍限所测单VE一步提示。' if metric['latency_reduction_percent']>0 else '正式对照未取得收益，保留负结果。'),
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T123247Z-sd-turbo-im2col-shapes.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        milestones.append({'report':'docs/impl/'+name,'title':'VAE im2col真实形状诊断',
                           'data':'10项CPU检查；40次/请求、11种VAE形状；逻辑输出19.391GiB/请求',
                           'summary':'512×512两种形状占约58%写入量；诊断轮不宣称性能收益。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T123747Z-sd-turbo-im2col-rows.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        milestones.append({'report':'docs/impl/'+name,'title':'im2col输出行划分候选',
                           'data':'16个VE按位边界用例通过；并发和性能待验证',
                           'summary':'输出行所有权顺序组合正确，尚未接入模型或证明提速。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T123938Z-sd-turbo-im2col-micro-timeout.json'
    if proof.exists():
        name='20261009T123938Z-sd-turbo-im2col-micro.md';p=ROOT/'docs/impl'/name
        milestones.append({'report':'docs/impl/'+name,'title':'im2col串行参考套件（超时）',
                           'data':'128通道26.400→10.111ms；23项通过后600秒超时124',
                           'summary':'128通道部分完整对照有收益，256通道未完成，不作为完整套件通过。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T125100Z-sd-turbo-im2col-parallel-reference.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        data='；'.join('%d通道%.3f→%.3fms（−%.2f%%）'%(m['ic'],m['channels_mean_seconds']*1000,m['rows_mean_seconds']*1000,m['latency_reduction_percent']) for m in v['metrics'])
        milestones.append({'report':'docs/impl/'+name,'title':'im2col输出行划分并发微基准',
                           'data':data+'；32次全元素检查通过',
                           'summary':'两种真实形状内核均受益，仍未接入模型或测量整图收益。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T125611Z-sd-turbo-im2col-concurrent.json'
    if proof.exists():
        name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        milestones.append({'report':'docs/impl/'+name,'title':'im2col并发与分区所有权验证',
                           'data':'40组所有权检查、32种实际并发组合通过；本轮未测性能',
                           'summary':'支持1/2/4/8实际线程；16分区仅顺序所有权验证，整图接入待执行。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T130018Z-sd-turbo-im2col-model.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        data='双请求冷/热%.3f/%.3fs；%d项CPU检查；同二进制收益未测'%(v['request_seconds'][0],v['request_seconds'][1],v['independent_cpu_checks'])
        milestones.append({'report':'docs/impl/'+name,'title':'im2col输出行模型接入（正确性轮）',
                           'data':data,'summary':'VAE派发8候选/32原节点通过；已完成'+','.join(v['completed_tests'])+('，正式对照见后续结果。' if any(json.loads(q.read_text()).get('earlier_proof_sha256')==hashlib.sha256(proof.read_bytes()).hexdigest() for q in (ROOT/'docs/results').glob('*-sd-turbo-im2col-abba.json')) else '，正式对照待完成。'),
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-im2col*abba.json')):
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name;m=v['metrics']['request1']
        extended='rows512_seconds' in m
        before_key,after_key=('rows512_seconds','rows256_seconds') if extended else ('channels_seconds','rows_seconds')
        data='热%.3f→%.3fs（延迟变化%+.2f%%）；%d项CPU检查，字节一致'%(m[before_key],m[after_key],-m['latency_reduction_percent'],v['total_cpu_stage_checks'])
        milestones.append({'report':'docs/impl/'+name,'title':('im2col扩大256范围整图对照' if extended else 'im2col输出行整图对照（'+v['model_kernel_optimization']['ve_sd_turbo_im2col_rows.c']+'接入）'),
                           'data':data,'summary':('当前构建退化不采用；发现候选遗漏O2，修正后需重新验证。' if v['model_kernel_optimization']['ve_sd_turbo_im2col_rows.c']=='-O1' else '同二进制对照确认收益，仍限固定单VE一步提示。' if m['latency_reduction_percent']>0 else '修正构建整图未获收益，保留负结果并继续诊断。'),
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T132236Z-sd-turbo-im2col-o2-model.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        extra=('；六提示一步热%.3f–%.3fs'%(min(v['six']['request_seconds'][1:]),max(v['six']['request_seconds'][1:])) if 'six' in v else '')+('；四步热%.3fs'%v['four']['request_seconds'][1] if 'four' in v else '')
        milestones.append({'report':'docs/impl/'+name,'title':'im2col O2模型构建与实际对象验证',
                           'data':('冷/热%.3f/%.3fs；%d项模型CPU检查；实际对象40组所有权/32种并发通过；收益未测'%(v['request_seconds'][0],v['request_seconds'][1],v['independent_cpu_checks']) if 'request_seconds' in v else '实际O2对象40组所有权/32种并发边界通过；性能未测')+extra,
                           'summary':'实际对象一致；已完成'+','.join(v.get('completed_tests',[]))+('；正确性轮不单独计算收益，正式对照另列。' if any(json.loads(q.read_text()).get('earlier_proof_sha256')==hashlib.sha256(proof.read_bytes()).hexdigest() for q in (ROOT/'docs/results').glob('*-sd-turbo-im2col-abba.json')) else '，正式整图收益待验证。'),
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T134511Z-sd-turbo-im2col-256.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        data='；'.join('%d通道%.3f→%.3fms（−%.2f%%）'%(m['ic'],m['channels_mean_seconds']*1000,m['rows_mean_seconds']*1000,m['latency_reduction_percent']) for m in v['metrics'])
        milestones.append({'report':'docs/impl/'+name,'title':'im2col 256形状实际对象微基准',
                           'data':data+'；32次完整按位检查通过',
                           'summary':'两种剩余真实形状内核受益，扩大派发后的整图收益待验证。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    proof=ROOT/'docs/results/20261009T134952Z-sd-turbo-im2col-256-model.json'
    if proof.exists():
        v=json.loads(proof.read_text());name=proof.name.replace('.json','.md');p=ROOT/'docs/impl'/name
        data='；'.join(r['name']+'冷/热'+('/'.join('%.3fs'%t for t in (r['request_seconds'][0],r['request_seconds'][-1]))) for r in v['tests'])
        milestones.append({'report':'docs/impl/'+name,'title':'im2col扩大256形状模型验证',
                           'data':data+'；%d项CPU检查；新增收益未测'%v['independent_cpu_checks'],
                           'summary':'实际15候选/25原派发通过；已完成'+','.join(v['completed_tests'])+'，正式对照结果另列。',
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-rows256-profile.json')):
        v=json.loads(proof.read_text())
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'扩大范围后算子诊断',
                           'data':'冷/热'+ '/'.join(f'{x:.3f}s' for x in v['request_seconds'])+'；UNet Softmax 0.592s/CONT 0.374s；VAE矩阵4.402s/im2col 0.482s；10项CPU检查',
                           'summary':'诊断包含探针开销，不计算收益；下一步分解VAE矩阵与线程恢复成本。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-rows256-nlc-profile.json')):
        v=json.loads(proof.read_text());r=v['requests'][1]
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'扩大范围后的逐矩阵诊断',
                           'data':f"热{r['request_seconds']:.3f}s；VAE40次矩阵{r['stages']['vae']['gemm_seconds']:.3f}s/阶段{r['stages']['vae']['backend_blas_seconds']:.3f}s；10项CPU检查",
                           'summary':'诊断不计算收益；按真实长空间维度测试分块候选。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-tiles.json')):
        v=json.loads(proof.read_text());r=next(x for x in v['comparisons'] if x['shape']==1 and x['tile']==16384)
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE长空间矩阵分块微基准',
                           'data':f"256通道{r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms（−{r['latency_reduction_percent']:.2f}%）；128通道全部退化；128次完整BLAS比较/2560独立FP64抽样",
                           'summary':'单形状内核收益成立，未接入模型；扩大其他真实形状验证后决定采用范围。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-extended.json')):
        v=json.loads(proof.read_text());r=[x for x in v['comparisons'] if x['tile']==8192 and x['shape']>0]
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE空间分块六形状扩展',
                           'data':'五种形状8192分块延迟降低'+ '/'.join(f"{x['latency_reduction_percent']:.2f}%" for x in r)+'；384完整BLAS比较/7680独立FP64抽样',
                           'summary':'128/262144/1152退化；仅将五种受益形状推进到模型验证，尚无整图收益。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']}
        data=str(v['independent_cpu_checks'])+'项CPU检查；'
        for name in ('double','six','four'):
            if name in tests:data+=name+'冷/末热'+ '/'.join(f'{x:.3f}s' for x in (tests[name]['request_seconds'][0],tests[name]['request_seconds'][-1]))+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE选择性空间分块模型验证','data':data+'正式收益另测','summary':'9个选择性节点/120分块调用通过，当前构建正确性验证完成。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE空间分块正式整图对照',
                           'data':f"热{hot['none']['mean_seconds']:.3f}→{hot['8192']['mean_seconds']:.3f}s（−{hot['latency_reduction_percent']:.2f}%）；142项CPU检查、F32/PNG字节一致；六热样本/模式",
                           'summary':'实际模型小幅收益确认，默认关闭；下一步拆分约1.16秒图像输出阶段。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-image-output-profile.json')):
        v=json.loads(proof.read_text());r=v['requests'][1]
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'图像输出子阶段诊断',
                           'data':f"热{r['request_seconds']:.3f}s；输出{r['image_output_seconds']:.3f}s；PNG0.783s/准备0.200s/范围0.079s/RGB0.110s；10项CPU检查",
                           'summary':'与此前正式候选字节一致，诊断无收益结论；下一步独立测试同参数PNG编码O2候选。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-png-o2.json')):
        v=json.loads(proof.read_text());r=v['real_six_mean']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'PNG隔离O2独立编码验证',
                           'data':f"六真实图像O1/O2编码{r['reference_seconds']:.3f}→{r['candidate_seconds']:.3f}s（−{r['latency_reduction_percent']:.2f}%）；240完整CPU解码/PNG字节对照",
                           'summary':'独立编码受益；原模型主程序实际O0，模型收益另测；两次构建失败已留档。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-png-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项CPU检查；'
        for name in ('double','six','four'):
            if name in tests:data+=name+'冷/末热'+ '/'.join(f'{x:.3f}s' for x in (tests[name]['request_seconds'][0],tests[name]['request_seconds'][-1]))+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'PNG隔离O2实际模型验证','data':data+'正式收益另测','summary':'实际PNG对象O2、主程序O0核验通过；候选多样正确性完成。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-png-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];image=v['hot_image_output_mean_seconds']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'PNG隔离O2正式整图对照',
                           'data':f"热{hot['generic']['mean_seconds']:.3f}→{hot['ve']['mean_seconds']:.3f}s（−{hot['latency_reduction_percent']:.2f}%）；输出{image['generic']:.3f}→{image['ve']:.3f}s；142项CPU检查、字节一致",
                           'summary':'固定热负载收益确认，含冷加载进程未改善；下一步验证RGB缓冲复用。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-rgb-buffer-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项CPU检查；'
        for name in ('double','six','four'):
            if name in tests:
                times=tests[name]['request_seconds'];data+=name+'冷'+f'{times[0]:.3f}s'+'、热'+','.join(f'{x:.3f}s' for x in times[1:])+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'RGB缓冲复用实际模型验证','data':data+'正式收益待对照','summary':'逐请求实际分配/复用与CPU参考核验；不同提示词检查覆盖缓冲残留风险，不据单次速度宣称提升。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-rgb-buffer-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];image=v['hot_image_output_mean_seconds']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'RGB缓冲复用正式整图对照',
                           'data':f"热{hot['request']['mean_seconds']:.3f}→{hot['resident']['mean_seconds']:.3f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；输出{image['request']:.3f}→{image['resident']:.3f}s；{v['independent_cpu_checks']}项CPU检查、字节一致",
                           'summary':'仅固定单VE一步热负载结论，冷加载及进程耗时另列，默认保持请求分配。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-o2.json')):
        v=json.loads(proof.read_text());clamp=v['real_six_mean']['clamp'];pack=v['real_six_mean']['pack']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'像素钳制与打包隔离O2独立验证',
                           'data':f"六CPU模型decoded输入：钳制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms，打包{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms；27组float/18组RGB完整CPU字节检查、288次钳制/打包计时",
                           'summary':'独立O0/O2核受益且字节一致，未接入模型；编译器报告未向量化，完整模型收益待测。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项新构建CPU检查；'
        for name in ('double','six','four'):
            if name in tests:
                times=tests[name]['request_seconds'];data+=name+'冷'+f'{times[0]:.3f}s'+'、热'+','.join(f'{x:.3f}s' for x in times[1:])+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'隔离O2像素核实际模型验证','data':data+'正式收益待对照',
                           'summary':'实际像素O2、主程序O0、PNG O2核验与模型参考验证；保持追踪边界，完整请求收益另测。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];image=v['hot_image_output_mean_seconds']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'隔离O2像素核正式整图对照',
                           'data':f"热{hot['generic']['mean_seconds']:.3f}→{hot['ve']['mean_seconds']:.3f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；输出{image['generic']:.3f}→{image['ve']:.3f}s；{v['independent_cpu_checks']}项CPU检查、字节一致",
                           'summary':'固定单VE一步热负载正式对照，冷请求与进程耗时另列；不外推多步或任意提示，原路径默认保留。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-vector.json')):
        v=json.loads(proof.read_text());clamp=v['real_six_mean']['clamp'];pack=v['real_six_mean']['pack']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'像素循环重排独立O2对照',
                           'data':f"六CPU模型decoded输入：钳制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms（−{clamp['latency_reduction_percent']:.2f}%），打包{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms（−{pack['latency_reduction_percent']:.2f}%）；33组float/18组RGB完整CPU字节检查",
                           'summary':'钳制计算循环向量化，有限值扫描/打包仍标量；独立核收益确认，模型收益未测量。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-scan.json')):
        v=json.loads(proof.read_text());clamp=v['real_six_mean']['clamp'];pack=v['real_six_mean']['pack']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'有限值扫描无条件归约候选',
                           'data':f"同期钳制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms（延迟变化{-clamp['latency_reduction_percent']:+.2f}%）；打包控制{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms；33组float/18组RGB完整CPU字节检查",
                           'summary':'扫描仍未向量化；退化候选拒绝，保留上一轮基线；不声称模型收益。' if clamp['latency_reduction_percent']<=0 else '独立扫描候选收益，模型收益另测；打包代码未变，不计为优化。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-any.json')):
        v=json.loads(proof.read_text());clamp=v['real_six_mean']['clamp'];pack=v['real_six_mean']['pack']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'异常标记归约与定位回退候选',
                           'data':f"同期钳制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms（延迟变化{-clamp['latency_reduction_percent']:+.2f}%）；打包控制{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms；33组float/18组RGB完整CPU字节检查",
                           'summary':'异常扫描仍未向量化，退化候选拒绝；实际对象保留有限值包装调用，下一步核验无函数调用分类。' if clamp['latency_reduction_percent']<=0 else '独立核收益，模型收益另测；打包代码未变，不计为优化。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-bits.json')):
        v=json.loads(proof.read_text());clamp=v['real_six_mean']['clamp'];pack=v['real_six_mean']['pack']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'位模式有限值分类独立候选',
                           'data':f"同期钳制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms（延迟降低{clamp['latency_reduction_percent']:.2f}%）；打包控制{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms；{v['full_cpu_float_checks']}组float/{v['full_cpu_rgb_checks']}组RGB完整CPU字节检查",
                           'summary':'扫描部分向量化且实际对象无有限值包装调用；独立核收益确认，下一步接入模型，打包未计为收益。' if clamp['latency_reduction_percent']>0 else '独立候选未获收益，保留负结果；模型未接入，打包为控制组。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-bits-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项当前模型CPU检查；'
        for name in ('double','six','four'):
            if name in tests:
                times=tests[name]['request_seconds'];data+=name+'冷'+f'{times[0]:.3f}s'+'、热'+','.join(f'{x:.3f}s' for x in times[1:])+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'位模式像素核实际模型验证','data':data+'增量收益待交替对照',
                           'summary':'实际O2位模式核与CPU参考/PNG字节验证通过，正式整模型增量收益另测。' if v['all_candidate_tests_completed'] else '模型接入验证尚未完成，不宣称性能收益。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-bits-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];image=v['hot_image_output_mean_seconds'];process=v['metrics']['process']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'位模式像素核双构建正式增量对照',
                           'data':f"热{hot['baseline']['mean_seconds']:.3f}→{hot['candidate']['mean_seconds']:.3f}s（−{hot['latency_reduction_percent']:.2f}%）；输出{image['baseline']*1000:.3f}→{image['candidate']*1000:.3f}ms；进程{process['baseline']:.3f}→{process['candidate']:.3f}s；{v['independent_cpu_checks']}项CPU/36张PNG核验",
                           'summary':'固定热负载小幅增量收益，冷请求和进程未改善；双二进制链接布局不同，默认原路径保留。' if hot['latency_reduction_percent']>0 else '完整热请求未获收益，保留负结果；双二进制对照，不外推独立内核提升。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-pack.json')):
        v=json.loads(proof.read_text());pack=v['real_six_mean']['pack'];clamp=v['real_six_mean']['clamp']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'精确RGB打包独立候选',
                           'data':f"六真实输入打包{pack['reference_seconds']*1000:.3f}→{pack['candidate_seconds']*1000:.3f}ms（−{pack['latency_reduction_percent']:.2f}%）；钳制控制{clamp['reference_seconds']*1000:.3f}→{clamp['candidate_seconds']*1000:.3f}ms；43组float/32组RGB及255阈值核验",
                           'summary':'打包部分向量化且实际对象无round调用；独立核收益确认，完整模型未接入，钳制波动不计收益。' if pack['latency_reduction_percent']>0 else '独立打包未获收益，保留负结果；模型未接入，钳制是控制组。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-pack-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项当前模型CPU检查；'
        for name in ('double','six','four'):
            if name in tests:
                times=tests[name]['request_seconds'];data+=name+'冷'+f'{times[0]:.3f}s'+'、热'+','.join(f'{x:.3f}s' for x in times[1:])+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'精确RGB打包实际模型验证','data':data+'增量收益待交替对照',
                           'summary':'实际O2对象与独立测试对象一致，CPU/PNG及打包输入范围验证通过；完整模型新增收益另测。' if v['all_candidate_tests_completed'] else '精确打包模型验证尚未完成，不宣称性能收益。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-pixels-pack-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];image=v['hot_image_output_mean_seconds'];process=v['metrics']['process']
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'精确RGB打包双构建正式增量对照',
                           'data':f"热{hot['baseline']['mean_seconds']:.3f}→{hot['candidate']['mean_seconds']:.3f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；输出{image['baseline']*1000:.3f}→{image['candidate']*1000:.3f}ms；进程{process['baseline']:.3f}→{process['candidate']:.3f}s；{v['independent_cpu_checks']}项CPU/36张PNG核验",
                           'summary':'输出阶段改善；完整热请求均值差很小且配对方向相反，尚未证明稳定整模型收益，冷请求/进程略退化。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-shapes.json')):
        v=json.loads(proof.read_text());times=v['request_seconds'];hot=v['requests'][1]['stages'];shape_groups=v['hot_shape_groups']
        data=f"冷/热{times[0]:.3f}/{times[1]:.3f}s；CONT "+'，'.join(f"{stage} {hot[stage]['seconds']:.3f}s/{hot[stage]['nodes']}节点" for stage in ('clip','unet','vae'))+f"；{len(shape_groups)}组实际布局；10项CPU/2张PNG核验"
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'CONT实际形状与耗时诊断','data':data,
                           'summary':'诊断含节点计时/屏障开销，无新数据计算核或提速结论；依据真实布局选择下一拷贝候选。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-transpose.json')):
        v=json.loads(proof.read_text());data='；'.join(','.join(map(str,r['shape'][:3]))+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms（−{r['latency_reduction_percent']:.2f}%）" for r in v['results'])+'；180所有权/48并发/24实际图/14无效参数/6完整CPU输出'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'CONT转置拷贝独立图微基准','data':data,
                           'summary':'实际模型ggml CONT图基线与O2候选均按位通过，三真实布局独立图受益；模型未接入，完整请求新增收益另测。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-model.json')):
        v=json.loads(proof.read_text());tests={e['name']:e for e in v['tests']};data=str(v['independent_cpu_checks'])+'项当前模型CPU检查；'
        for name in ('double','six','four'):
            if name in tests:
                times=tests[name]['request_seconds'];data+=name+'冷'+f'{times[0]:.3f}s'+'、热'+','.join(f'{x:.3f}s' for x in times[1:])+'；'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'CONT转置核实际模型接入','data':data+'每步UNet15候选/243回退，正式收益待对照',
                           'summary':'首轮错误入口零派发已拒绝并修正；实际对象一致和CPU/PNG验证通过，完整请求新增收益另测。' if v['all_candidate_tests_completed'] else 'CONT实际模型接入验证尚未完成，不宣称性能收益。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];stages=v['hot_stage_profile_seconds'];process=v['metrics']['process']
        data=f"热{hot['generic']['mean_seconds']:.3f}→{hot['ve']['mean_seconds']:.3f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；UNet CPU阶段{stages['unet/backend_CPU']['generic']:.3f}→{stages['unet/backend_CPU']['ve']:.3f}s；进程{process['generic']:.3f}→{process['ve']:.3f}s；142项CPU/26张PNG核验"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='同二进制固定热负载测得增量收益，所有输出按位相同；冷请求与进程另列，不泛化提示或多步性能。' if consistent else '热均值与配对结果未共同证明稳定收益，保留全部数据；同二进制固定负载，不外推独立图提升。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'CONT转置核正式完整请求对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-expanded.json')):
        v=json.loads(proof.read_text());data='；'.join(','.join(map(str,r['shape'][:3]))+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms" for r in v['results'])+'；216计时图/270所有权/72并发/72图检查/14无效参数/18CPU输出'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'九种新增CONT布局独立图验证','data':data,'summary':'九实际布局独立图均获益且按位正确；模型未扩展，完整请求新增收益另测。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-extended-model.json')):
        v=json.loads(proof.read_text());tests={r['name']:r for r in v['tests']};data='；'.join(name+'冷'+f"{tests[name]['request_seconds'][0]:.3f}s、热"+','.join(f"{x:.3f}s" for x in tests[name]['request_seconds'][1:]) for name in ('double','six','four'))+'；62CPU/10PNG；CLIP23、UNet45/步、VAE3候选'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'十二CONT布局实际模型验证','data':data,'summary':'九新增布局按阶段白名单实际命中，全部输出与前版按位相同；完整请求新增收益待同程序对照。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-cont-extended-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.3f}→{hot['extended']['mean_seconds']:.3f}s（−{hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.3f}→{cold['extended']:.3f}s；进程{process['original']:.3f}→{process['extended']:.3f}s；142CPU/26PNG"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='新增九布局同程序固定热请求收益通过，输出按位相同；冷请求和进程另列，不泛化其他输入或多步性能。' if consistent else '正式均值及配对未共同证明稳定收益，保留全部数据，不外推独立算子结果。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'十二CONT布局完整请求正式对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-current-profile.json')):
        v=json.loads(proof.read_text());times=v['request_seconds'];soft=next(x['seconds'] for x in v['hot_stages']['unet']['operators'] if x['op']=='SOFT_MAX');blas=sum(x['seconds'] for x in v['hot_component_profile'] if x['stage']=='vae' and x['part']=='backend_BLAS')
        data=f"冷/热{times[0]:.3f}/{times[1]:.3f}s；UNet Softmax{soft:.3f}s，VAE矩阵{blas:.3f}s；10CPU/2PNG"
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'十二布局模型当前算子诊断','data':data,'summary':'保持模型且输出按位不变；诊断有计时开销，无新增收益，依据当前数据推进Softmax和VAE候选。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-shapes.json')):
        v=json.loads(proof.read_text());times=v['request_seconds'];shapes=v['shape_requests'][1]['shapes'];n=len({(r['stage'],tuple(r['ne'])) for r in shapes})
        data=f"冷/热{times[0]:.3f}/{times[1]:.3f}s；{len(shapes)}调用/{n}形状/{len(v['shape_requests'][1]['phases'])}线程记录；列宽64/77/256/1024/4096；10CPU/2PNG"
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'Softmax实际形状及逐行阶段诊断','data':data,'summary':'输出按位正确，逐行计时开销明显，不能推断正常阶段占比或新增收益；下一步低开销批量测量。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-sum-o2.json')):
        v=json.loads(proof.read_text());data='；'.join(str(r['columns'])+f"列 {r['baseline_seconds']*1e6:.6f}→{r['candidate_seconds']*1e6:.6f}µs（延迟降低{r['latency_reduction_percent']:.4f}%）" for r in v['results'])+'；108CPU按位/120批'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'严格顺序FP64求和O2批量对照','data':data,'summary':'短列收益很小、长列几乎不变；保持模型，不接入，不宣称完整Softmax或请求收益。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-scale.json')):
        v=json.loads(proof.read_text());data='；'.join(str(r['columns'])+'列 '+r['operation']+f" {r['baseline_seconds']*1e6:.6f}→{r['candidate_seconds']*1e6:.6f}µs" for r in v['results'])+'；144VE输出对/288带符号清零CPU输出/240批；严格IEEE失败'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'Softmax复制缩放独立批量对照','data':data,'summary':'独立算子获益且保持VE基线位模式；严格IEEE参考因次正规清零失败，模型未改，实际ggml图和请求收益待验证。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-graph.json')):
        v=json.loads(proof.read_text());data='；'.join(','.join(map(str,r['shape'][:3]))+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms" for r in v['results'])+'；180多线程/45CPU/240图'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'实际Softmax图复制缩放对照','data':data,'summary':'七几何双侧获益，三几何不稳定或退化；实际原路径按位相同，CPU参考明确次正规清零，模型收益待验证。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-model.json')):
        v=json.loads(proof.read_text())
        if not v.get('all_candidate_tests_completed'):continue
        tests={r['name']:r for r in v['tests']};data='；'.join(name+'冷'+f"{tests[name]['request_seconds'][0]:.3f}s、热"+','.join(f"{x:.3f}s" for x in tests[name]['request_seconds'][1:]) for name in ('double','six','four'))+'；62CPU/10PNG；UNet30/32、VAE1/1候选'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'七种Softmax几何实际模型验证','data':data,'summary':'选择性缩放实际命中，完整输出与前版按位相同；尚无同程序完整请求新增收益结论。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-softmax-scale-abba.json')):
        v=json.loads(proof.read_text());hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.3f}→{hot['candidate']['mean_seconds']:.3f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.3f}→{cold['candidate']:.3f}s；进程{process['original']:.3f}→{process['candidate']:.3f}s；142CPU/26PNG"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='七形状Softmax缩放同程序固定热请求均值及双侧配对获益，输出按位相同；冷态/进程单列，不泛化其他负载。' if consistent else '完整热请求均值及双侧配对未共同证明稳定收益，保留退化及全部样本；独立算子收益不外推。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'Softmax选择性缩放完整请求正式对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-shapes.json')):
        v=json.loads(proof.read_text());times=v['request_seconds'];shapes=v['shape_requests'][1]['shapes'];n=len({(r['stage'],tuple(r['ne']),r['groups'],r['epsilon']) for r in shapes});unet=next(r['seconds'] for r in v['hot_stages']['unet']['operators'] if r['op']=='GROUP_NORM');vae=next(r['seconds'] for r in v['hot_stages']['vae']['operators'] if r['op']=='GROUP_NORM')
        data=f"诊断冷/热{times[0]:.3f}/{times[1]:.3f}s；GroupNorm UNet{unet:.6f}s/61节点、VAE{vae:.6f}s/30节点；{n}几何/epsilon组合；10CPU/2PNG"
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm真实几何及当前算子诊断','data':data,'summary':'两epsilon及24组合以实际采集为准，输出与前版按位一致；诊断无新增收益，下一步保持求和顺序做组内长向量缩放。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-scale.json')):
        v=json.loads(proof.read_text());data='；'.join(str(r['id'])+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms" for r in v['results'])+'；288VE/24CPU/576图'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm实际图按组长向量缩放','data':data,'summary':'按位和独立CPU参考通过；短图波动、多个退化且最大VAE几乎不变，不接入模型，先降低诊断计时干扰做批量复测。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-batched.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='actual_group_norm_batched_verified' or not v.get('diagnostic_only_delta_verified'):continue
        data='；'.join(str(r['id'])+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms" for r in v['results'])+'；288VE/24CPU；576批×16图，计时静默'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm静默批量实际图对照','data':data,'summary':'保持数学对象与逐行求和顺序，正确性阶段核验实际团队，静默批量结果逐配置判断；独立图结果不外推完整请求。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-center.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='actual_group_norm_center_verified' or not v.get('center_only_delta_verified'):continue
        data='；'.join(str(r['id'])+f" {r['baseline_seconds']*1000:.3f}→{r['candidate_seconds']*1000:.3f}ms" for r in v['results'])+'；288VE/24CPU；576批×16图；24明确候选捕获'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm中心化平方独立实际图对照','data':data,'summary':'保留均值及逐行FP64方差累加，中心化平方向量化；宽32及以上组合受益，宽8/16退化，选择性模型接入及完整收益待验证。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-model.json')):
        v=json.loads(proof.read_text())
        if not v.get('all_candidate_tests_completed') or not v.get('group_norm_actual_object_equal_independent'):continue
        tests={r['name']:r for r in v['tests']}
        data='；'.join(name+'冷'+f"{tests[name]['request_seconds'][0]:.3f}s、热"+','.join(f"{x:.3f}s" for x in tests[name]['request_seconds'][1:]) for name in ('double','six','four'))+'；62CPU/10PNG；UNet缩放61/61、中心化31/61，VAE30/30'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm选择性实际模型验证','data':data,'summary':'已测缩放与16中心化平方组合实际命中，新O2对象与独立测试一致，前版输出按位相同；完整请求增量收益另做同程序对照。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-group-norm-abba.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='formal_group_norm_abba_verified' or not v.get('cpu_validation_temperature'):continue
        hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.6f}→{hot['candidate']['mean_seconds']:.6f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.6f}→{cold['candidate']:.6f}s；进程{process['original']:.6f}→{process['candidate']:.6f}s；142CPU/26PNG"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='同程序仅切换GroupNorm，固定热请求均值及双侧配对均获益，输出按位相同；冷/进程单列，不外推其他输入或多步。' if consistent else '热请求均值与双侧配对未共同证明稳定收益，保留全部样本和退化；独立图收益不外推完整请求。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'GroupNorm完整请求同程序正式对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-small.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature'):continue
        data='；'.join(str(r['shape'])+'/'+str(r['tile'])+f" {r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms" for r in v['comparisons'])+'；基线shape0=0、其余8192；288完整BLAS/5760FP64抽样'
        stable=[r for r in v['comparisons'] if r['both_pairs_positive'] and r['latency_reduction_percent']>0]
        summary='独立六形状、4线程、小分块相对当前基线逐配置筛选，保留退化与双侧不一致；尚无模型或整图增量收益。' if stable else '所有较小分块均未证明稳定收益，保持当前模型配置；独立矩阵结果不外推完整请求。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE较小空间分块相对当前基线筛选','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-small-model.json')):
        v=json.loads(proof.read_text())
        if not v.get('all_candidate_tests_completed') or not v.get('cpu_validation_temperature') or not v.get('previous_model_byte_comparison',{}).get('all_trace_and_png_bytes_identical'):continue
        tests={r['name']:r for r in v['tests']}
        data='；'.join(name+'冷'+f"{tests[name]['request_seconds'][0]:.3f}s、热"+','.join(f"{x:.3f}s" for x in tests[name]['request_seconds'][1:]) for name in ('double','six','four'))+'；62CPU/10PNG；9节点/168分块调用；仅两形状4096，其余8192'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE按形状小分块实际模型验证','data':data,'summary':'只有BLAS对象及对应静态库变化，其余对象不变，实际派发与前版输出按位相同；正确性轮不认领完整请求增量收益，后续同程序对照。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-small-abba.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='formal_small_spatial_abba_verified' or not v.get('cpu_validation_temperature'):continue
        hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.6f}→{hot['candidate']['mean_seconds']:.6f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.6f}→{cold['candidate']:.6f}s；进程{process['original']:.6f}→{process['candidate']:.6f}s；142CPU/26PNG；8192/mixed4096"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='同程序只切换两个VAE形状分块，固定热请求均值及双侧配对获益，输出按位相同；冷/进程单列，不外推其他输入或多步。' if consistent else '热请求均值与双侧配对未共同证明稳定收益，保留全部样本及退化；独立矩阵收益不外推完整请求。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE按形状小分块完整请求正式对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-untiled.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature'):continue
        data='；'.join(str(r['shape'])+'/'+str(r['tile'])+f" {r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms" for r in v['comparisons'])+'；完整矩阵基线；96完整BLAS/1920FP64抽样'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'未分块VAE真实矩阵独立筛选','data':data,'summary':'512×16384×4608选择2048分块，两侧配对均获益；512×4096×4608全部退化保持完整矩阵；实际模型及整图增量收益待验证。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-untiled-model.json')):
        v=json.loads(proof.read_text())
        if not v.get('all_candidate_tests_completed') or not v.get('cpu_validation_temperature') or not v.get('pre_run_cooling') or not v.get('previous_model_byte_comparison',{}).get('all_trace_and_png_bytes_identical'):continue
        tests={r['name']:r for r in v['tests']}
        data='；'.join(name+'冷'+f"{tests[name]['request_seconds'][0]:.3f}s、热"+','.join(f"{x:.3f}s" for x in tests[name]['request_seconds'][1:]) for name in ('double','six','four'))+'；62CPU/10PNG；16节点/224分块调用；mixed2048'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'新增单形状2048分块实际模型验证','data':data,'summary':'保留mixed4096，仅新增一个受益形状；实际派发、对象增量和前版输出按位一致，完整请求增量收益待同程序对照。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-untiled-abba.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='formal_untiled_spatial_abba_verified' or not v.get('cpu_validation_temperature'):continue
        hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.6f}→{hot['candidate']['mean_seconds']:.6f}s（延迟降低{hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.6f}→{cold['candidate']:.6f}s；进程{process['original']:.6f}→{process['candidate']:.6f}s；142CPU/26PNG；mixed4096/mixed2048"
        consistent=hot['latency_reduction_percent']>0 and all(p['latency_reduction_percent']>0 for p in v['paired_hot_arm_means'])
        summary='同程序只切换新增VAE形状分块，固定热请求均值及双侧配对获益，输出按位相同；样本少、冷/进程单列，不外推其他输入或多步。' if consistent else '热请求均值与双侧配对未共同证明收益，保留全部样本及退化；独立矩阵收益不外推完整请求。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'新增2048形状完整请求正式对照','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-current-vae-threads.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='model_validation_verified' or not v.get('cpu_validation_temperature') or not v.get('pre_run_cooling'):continue
        tests={e['name']:e for e in v['tests']}
        double=tests['double']['request_seconds'];six=tests['six']['request_seconds'];four=tests['four']['request_seconds']
        data=f"双请求冷/热{double[0]:.6f}/{double[1]:.6f}s；六提示冷{six[0]:.6f}s、热"+'/'.join(f"{x:.6f}" for x in six[1:])+f"s；四步冷/热{four[0]:.6f}/{four[1]:.6f}s；62CPU/10PNG；初始覆盖失败83.995850/9.889924s保留"
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'当前优化模型VAE整阶段四线程验证','data':data,'summary':'先验证真实四线程输出行对象，再只扩展调用条件至4/8；完整模型输出与前版按位相同，正确性通过，尚未证明整请求性能收益，下一轮同二进制交替对照。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-current-vae-abba.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='formal_current_vae_thread_abba_verified' or not v.get('cpu_validation_temperature'):continue
        hot=v['hot_requests'];cold=v['metrics']['request0'];process=v['metrics']['process']
        data=f"热{hot['original']['mean_seconds']:.6f}→{hot['candidate']['mean_seconds']:.6f}s（延迟增加{-hot['latency_reduction_percent']:.2f}%）；冷{cold['original']:.6f}→{cold['candidate']:.6f}s；进程{process['original']:.6f}→{process['candidate']:.6f}s；142CPU/26PNG；通用8/矩阵4与整阶段4"
        summary='同程序只切换VAE线程策略，整阶段4热均值、双侧配对及全部热序号退化；输出按位相同，保留通用8/矩阵4，下一步独立筛选输出通道分块。'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE整阶段四线程正式对照（拒绝）','data':data,'summary':summary})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-channels.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature'):continue
        data='；'.join(f"{r['shape']}/{r['tile']} {r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms" for r in v['comparisons'])+'；基线空间块2048/8192/完整；96完整BLAS/1920FP64抽样'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE输出通道分块独立筛选（拒绝）','data':data,'summary':'三个真实矩阵的128/256输出通道分块均在均值和双侧配对退化，全部拒绝；当前模型保持不变，下一步细化已有空间分块。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-spatial-refine.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature'):continue
        data='；'.join(f"{r['shape']}/{r['tile']} {r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms" for r in v['comparisons'])+'；基线2048/4096/4096；144完整BLAS/2880FP64抽样'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE空间分块邻近值筛选','data':data,'summary':'八候选退化，第三形状3072块仅约0.056%正向，不据此认领完整模型改善，保持现有配置；下一步测试矩阵预排列并计入总成本。'})
    for proof in sorted((ROOT/'docs/results').glob('*-sd-turbo-gemm-pretranspose.json')):
        v=json.loads(proof.read_text())
        if v.get('status')!='current_baseline_microbenchmark_verified' or not v.get('cpu_validation_temperature'):continue
        data='；'.join(f"{r['m']}/{r['n']}/{r['k']} {r['baseline_mean_seconds']*1000:.3f}→{r['candidate_mean_seconds']*1000:.3f}ms" for r in v['comparisons'])+'；含排列/准备/释放；48完整BLAS/960FP64；完整最大差值0.000450134277'
        milestones.append({'report':'docs/impl/'+proof.stem+'.md','title':'VAE矩阵输入预排列含全部成本筛选','data':data,'summary':'三形状总成本双侧获益45.65%–56.29%，输入与边界通过但输出非按位相同；先实际模型精度与图像验证，再确认完整请求收益，不提前采用。'})
    milestones.sort(key=lambda item:Path(item['report']).name[:16])
    overrides={m['report']:m['summary'] for m in milestones}
    iterations=[]
    for timestamp,paths in groups.items():
        p=next((p for p in paths if p.name.endswith('-report.md')),paths[0])
        text=re.sub(r'<!-- OPTIMIZATION_HISTORY_START -->.*?<!-- OPTIMIZATION_HISTORY_END -->','',p.read_text(),flags=re.S);title=re.sub(r'^#\s*','',text.splitlines()[0])
        title=re.sub(r'^迭代\s*\d+[：:]\s*','',title)
        report=str(p.relative_to(ROOT))
        summary=overrides.get(report,source_summary(text,title))
        iterations.append({'timestamp':timestamp,'report':report,'related_reports':[str(q.relative_to(ROOT)) for q in paths],
                           'title':title,'data_excerpt':source_excerpt(text),'summary':summary,
                           'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    value={'current_report':str(current.relative_to(ROOT)),'next_direction':args.next_direction,
           'milestones':milestones,'all_iterations':iterations,
           'scope':'curated performance/negative/diagnostic milestones plus all public implementation iterations; index excerpts are historical report snapshots, not new measurements',
           'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def render(base):
        def link(report):
            p=Path(report);return ('../impl/'+p.name) if base=='guide' else p.name
        lines=['# 累计优化结果与下一步','',
               '本表同时保留成功、退化、失败和待验证结果。不同模型、输入、冷/热请求、单层/整图计时不可混算，各轮百分比不能相加或连乘；详细正确性、温度与风扇记录以链接报告为准。','',
               '当前轮：[报告]('+link(value['current_report'])+')。下一步：'+args.next_direction,'',
               '## 关键优化结果（全部已整理性能/负结果及当前轮）','',
               '| 轮次/模型 | 数据 | 一句话总结 |','| --- | --- | --- |']
        for m in milestones:lines.append('| ['+cell(m['title'])+']('+link(m['report'])+') | '+cell(m['data'])+' | '+cell(m['summary'])+' |')
        lines += ['', '## 全部迭代记录（含实施、修复、诊断及验证阶段）','',
                  '共%d个UTC时间戳迭代；同时间戳的实现/报告合并显示。数据列为原报告关键段落摘录，保留当时的失败/待验证措辞；它不将历史预测、CPU参考、独立算子或阶段性状态当作当前完整模型成绩。上方主表列出已整理的优化结论。'%len(iterations),'',
                  '| UTC轮次/报告 | 当轮数据摘录 | 一句话总结 |','| --- | --- | --- |']
        for row in iterations:lines.append('| ['+row['timestamp']+' '+cell(row['title'])+']('+link(row['report'])+') | '+cell(row['data_excerpt'])+' | '+cell(row['summary'])+' |')
        return '\n'.join(lines)+'\n'
    history_lines=['<!-- OPTIMIZATION_HISTORY_START -->','## 当前及此前各轮优化结果','',
                   '| 轮次/模型 | 数据 | 一句话总结 |','| --- | --- | --- |']
    for m in milestones:
        history_lines.append('| ['+cell(m['title'])+']('+Path(m['report']).name+') | '+cell(m['data'])+' | '+cell(m['summary'])+' |')
    history_lines += ['', '全部'+str(len(iterations))+'个时间戳迭代、包括实现/失败/待验证记录，见[本轮历史快照]('+stamp+'-optimization-history.md)。', '', '下一步：'+args.next_direction, '<!-- OPTIMIZATION_HISTORY_END -->']
    history_block='\n'.join(history_lines)+'\n'
    text=current.read_text()
    if '<!-- OPTIMIZATION_HISTORY_START -->' in text:
        text=re.sub(r'<!-- OPTIMIZATION_HISTORY_START -->.*?<!-- OPTIMIZATION_HISTORY_END -->\n?',lambda _:history_block,text,flags=re.S)
    else:
        text+='\n'+history_block
    current.write_text(text)
    for row in milestones+iterations:
        row['report_sha256']=hashlib.sha256((ROOT/row['report']).read_bytes()).hexdigest()
    output=ROOT/'docs/guides/optimization-history.md';output.write_text(render('guide'))
    snapshot=ROOT/'docs/impl'/(stamp+'-optimization-history.md');snapshot.write_text(render('snapshot'))
    machine=ROOT/'docs/results/optimization-history.json';machine.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'docs/results'/(stamp+'-optimization-history.json')).write_bytes(machine.read_bytes())
    print('History published:',len(milestones),'milestones;',len(iterations),'iterations')


if __name__=='__main__':main()
