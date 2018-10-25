from rx import Observable

numbers = Observable.from_([1,1,1,0,0,2,2,3]) \
    .scan(lambda x,i: max(x,i)) \
    .distinct() \
    .to_list() \
    .subscribe(lambda s: print(len(s)-1))
