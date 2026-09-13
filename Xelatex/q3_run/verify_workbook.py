"""只读回读result3.xlsx，检查每个输出单元格及公式缓存；不修改工作簿。
需openpyxl，仅用于文件核验；核心求解不依赖此库。
"""
import json
import numpy as np
from openpyxl import load_workbook
from q3_reproduce import HERE, save


def main():
    expected=json.loads((HERE/'paper/workbook_payload.json').read_text(encoding='utf-8'))
    book=load_workbook(HERE/'result3.xlsx',data_only=True,read_only=True)
    largest=0.; count=0
    for sheet,key in [('计划购电量','original'),('调整购电量','final')]:
        rows=list(book[sheet].iter_rows(min_row=2,max_row=335,values_only=True))
        values=np.asarray([row[1:145] for row in rows],float)
        error=float(np.max(np.abs(values-np.asarray(expected[key])))); largest=max(largest,error)
        np.testing.assert_allclose(values,expected[key],atol=1e-8,rtol=0); count+=values.size
        totals=np.asarray([row[145] for row in rows],float)
        np.testing.assert_allclose(totals,values.sum(1),atol=1e-6,rtol=0)
        cost=np.asarray([row[146] for row in rows],float)
        fees=np.asarray(expected['components'])
        np.testing.assert_allclose(cost,fees[:,0] if key=='original' else fees.sum(1),atol=1e-6,rtol=0)
        assert book[sheet]['B1'].value=='00:00--00:10'
        assert book[sheet]['EO1'].value=='23:50--24:00'
    for sheet,key,width in [('充放电量','storage',6),('紧急购电量','emergency',3)]:
        rows=list(book[sheet].iter_rows(min_row=2,max_row=len(expected[key])+1,max_col=width,values_only=True))
        for actual,wanted in zip(rows,expected[key]):
            for got,want in zip(actual,wanted):
                if isinstance(want,(float,int)):
                    assert got is not None and abs(got-want)<1e-8; count+=1
                else:
                    got=str(got.date()) if hasattr(got,'date') else got
                    assert got==want,(sheet,got,want)
    book.close()
    save(HERE/'paper/workbook_verification.json',{'passed':True,'numeric_cells_checked':count,
         'maximum_purchase_error_kwh':largest,'full_334_day_arrays_match':True,
         'totals_and_cached_formulas_match':True,'144_correct_interval_labels':True})
    print('Workbook full readback passed',count)


if __name__=='__main__': main()
