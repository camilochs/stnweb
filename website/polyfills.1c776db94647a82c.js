"use strict";(self.webpackChunkweb=self.webpackChunkweb||[]).push([[429],{955:(fe,ge,me)=>{const we="undefined"!=typeof globalThis&&globalThis,Me="undefined"!=typeof window&&window,Pe="undefined"!=typeof self&&"undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope&&self,Re=we||"undefined"!=typeof global&&global||Me||Pe,Te=function($,...O){if(Te.translate){const Y=Te.translate($,O);$=Y[0],O=Y[1]}let se=oe($[0],$.raw[0]);for(let Y=1;Y<$.length;Y++)se+=O[Y-1]+oe($[Y],$.raw[Y]);return se};function oe($,O){return":"===O.charAt(0)?$.substring(function ye($,O){for(let se=1,Y=1;se<$.length;se++,Y++)if("\\"===O[Y])Y++;else if(":"===$[se])return se;throw new Error(`Unterminated $localize metadata block in "${O}".`)}($,O)+1):$}Re.$localize=Te,me(583)},583:()=>{!function(e){const n=e.performance;function i(M){n&&n.mark&&n.mark(M)}function o(M,E){n&&n.measure&&n.measure(M,E)}i("Zone");const c=e.__Zone_symbol_prefix||"__zone_symbol__";function a(M){return c+M}const y=!0===e[a("forceDuplicateZoneCheck")];if(e.Zone){if(y||"function"!=typeof e.Zone.__symbol__)throw new Error("Zone already loaded.");return e.Zone}let d=(()=>{class M{constructor(t,r){this._parent=t,this._name=r?r.name||"unnamed":"<root>",this._properties=r&&r.properties||{},this._zoneDelegate=new v(this,this._parent&&this._parent._zoneDelegate,r)}static assertZonePatched(){if(e.Promise!==le.ZoneAwarePromise)throw new Error("Zone.js has detected that ZoneAwarePromise `(window|global).Promise` has been overwritten.\nMost likely cause is that a Promise polyfill has been loaded after Zone.js (Polyfilling Promise api is not necessary when zone.js is loaded. If you must load one, do so before loading zone.js.)")}static get root(){let t=M.current;for(;t.parent;)t=t.parent;return t}static get current(){return U.zone}static get currentTask(){return ae}static __load_patch(t,r,k=!1){if(le.hasOwnProperty(t)){if(!k&&y)throw Error("Already loaded patch: "+t)}else if(!e["__Zone_disable_"+t]){const C="Zone:"+t;i(C),le[t]=r(e,M,X),o(C,C)}}get parent(){return this._parent}get name(){return this._name}get(t){const r=this.getZoneWith(t);if(r)return r._properties[t]}getZoneWith(t){let r=this;for(;r;){if(r._properties.hasOwnProperty(t))return r;r=r._parent}return null}fork(t){if(!t)throw new Error("ZoneSpec required!");return this._zoneDelegate.fork(this,t)}wrap(t,r){if("function"!=typeof t)throw new Error("Expecting function got: "+t);const k=this._zoneDelegate.intercept(this,t,r),C=this;return function(){return C.runGuarded(k,this,arguments,r)}}run(t,r,k,C){U={parent:U,zone:this};try{return this._zoneDelegate.invoke(this,t,r,k,C)}finally{U=U.parent}}runGuarded(t,r=null,k,C){U={parent:U,zone:this};try{try{return this._zoneDelegate.invoke(this,t,r,k,C)}catch(J){if(this._zoneDelegate.handleError(this,J))throw J}}finally{U=U.parent}}runTask(t,r,k){if(t.zone!=this)throw new Error("A task can only be run in the zone of creation! (Creation: "+(t.zone||Q).name+"; Execution: "+this.name+")");if(t.state===x&&(t.type===te||t.type===P))return;const C=t.state!=p;C&&t._transitionTo(p,j),t.runCount++;const J=ae;ae=t,U={parent:U,zone:this};try{t.type==P&&t.data&&!t.data.isPeriodic&&(t.cancelFn=void 0);try{return this._zoneDelegate.invokeTask(this,t,r,k)}catch(l){if(this._zoneDelegate.handleError(this,l))throw l}}finally{t.state!==x&&t.state!==h&&(t.type==te||t.data&&t.data.isPeriodic?C&&t._transitionTo(j,p):(t.runCount=0,this._updateTaskCount(t,-1),C&&t._transitionTo(x,p,x))),U=U.parent,ae=J}}scheduleTask(t){if(t.zone&&t.zone!==this){let k=this;for(;k;){if(k===t.zone)throw Error(`can not reschedule task to ${this.name} which is descendants of the original zone ${t.zone.name}`);k=k.parent}}t._transitionTo(z,x);const r=[];t._zoneDelegates=r,t._zone=this;try{t=this._zoneDelegate.scheduleTask(this,t)}catch(k){throw t._transitionTo(h,z,x),this._zoneDelegate.handleError(this,k),k}return t._zoneDelegates===r&&this._updateTaskCount(t,1),t.state==z&&t._transitionTo(j,z),t}scheduleMicroTask(t,r,k,C){return this.scheduleTask(new g(L,t,r,k,C,void 0))}scheduleMacroTask(t,r,k,C,J){return this.scheduleTask(new g(P,t,r,k,C,J))}scheduleEventTask(t,r,k,C,J){return this.scheduleTask(new g(te,t,r,k,C,J))}cancelTask(t){if(t.zone!=this)throw new Error("A task can only be cancelled in the zone of creation! (Creation: "+(t.zone||Q).name+"; Execution: "+this.name+")");t._transitionTo(G,j,p);try{this._zoneDelegate.cancelTask(this,t)}catch(r){throw t._transitionTo(h,G),this._zoneDelegate.handleError(this,r),r}return this._updateTaskCount(t,-1),t._transitionTo(x,G),t.runCount=0,t}_updateTaskCount(t,r){const k=t._zoneDelegates;-1==r&&(t._zoneDelegates=null);for(let C=0;C<k.length;C++)k[C]._updateTaskCount(t.type,r)}}return M.__symbol__=a,M})();const w={name:"",onHasTask:(M,E,t,r)=>M.hasTask(t,r),onScheduleTask:(M,E,t,r)=>M.scheduleTask(t,r),onInvokeTask:(M,E,t,r,k,C)=>M.invokeTask(t,r,k,C),onCancelTask:(M,E,t,r)=>M.cancelTask(t,r)};class v{constructor(E,t,r){this._taskCounts={microTask:0,macroTask:0,eventTask:0},this.zone=E,this._parentDelegate=t,this._forkZS=r&&(r&&r.onFork?r:t._forkZS),this._forkDlgt=r&&(r.onFork?t:t._forkDlgt),this._forkCurrZone=r&&(r.onFork?this.zone:t._forkCurrZone),this._interceptZS=r&&(r.onIntercept?r:t._interceptZS),this._interceptDlgt=r&&(r.onIntercept?t:t._interceptDlgt),this._interceptCurrZone=r&&(r.onIntercept?this.zone:t._interceptCurrZone),this._invokeZS=r&&(r.onInvoke?r:t._invokeZS),this._invokeDlgt=r&&(r.onInvoke?t:t._invokeDlgt),this._invokeCurrZone=r&&(r.onInvoke?this.zone:t._invokeCurrZone),this._handleErrorZS=r&&(r.onHandleError?r:t._handleErrorZS),this._handleErrorDlgt=r&&(r.onHandleError?t:t._handleErrorDlgt),this._handleErrorCurrZone=r&&(r.onHandleError?this.zone:t._handleErrorCurrZone),this._scheduleTaskZS=r&&(r.onScheduleTask?r:t._scheduleTaskZS),this._scheduleTaskDlgt=r&&(r.onScheduleTask?t:t._scheduleTaskDlgt),this._scheduleTaskCurrZone=r&&(r.onScheduleTask?this.zone:t._scheduleTaskCurrZone),this._invokeTaskZS=r&&(r.onInvokeTask?r:t._invokeTaskZS),this._invokeTaskDlgt=r&&(r.onInvokeTask?t:t._invokeTaskDlgt),this._invokeTaskCurrZone=r&&(r.onInvokeTask?this.zone:t._invokeTaskCurrZone),this._cancelTaskZS=r&&(r.onCancelTask?r:t._cancelTaskZS),this._cancelTaskDlgt=r&&(r.onCancelTask?t:t._cancelTaskDlgt),this._cancelTaskCurrZone=r&&(r.onCancelTask?this.zone:t._cancelTaskCurrZone),this._hasTaskZS=null,this._hasTaskDlgt=null,this._hasTaskDlgtOwner=null,this._hasTaskCurrZone=null;const k=r&&r.onHasTask;(k||t&&t._hasTaskZS)&&(this._hasTaskZS=k?r:w,this._hasTaskDlgt=t,this._hasTaskDlgtOwner=this,this._hasTaskCurrZone=E,r.onScheduleTask||(this._scheduleTaskZS=w,this._scheduleTaskDlgt=t,this._scheduleTaskCurrZone=this.zone),r.onInvokeTask||(this._invokeTaskZS=w,this._invokeTaskDlgt=t,this._invokeTaskCurrZone=this.zone),r.onCancelTask||(this._cancelTaskZS=w,this._cancelTaskDlgt=t,this._cancelTaskCurrZone=this.zone))}fork(E,t){return this._forkZS?this._forkZS.onFork(this._forkDlgt,this.zone,E,t):new d(E,t)}intercept(E,t,r){return this._interceptZS?this._interceptZS.onIntercept(this._interceptDlgt,this._interceptCurrZone,E,t,r):t}invoke(E,t,r,k,C){return this._invokeZS?this._invokeZS.onInvoke(this._invokeDlgt,this._invokeCurrZone,E,t,r,k,C):t.apply(r,k)}handleError(E,t){return!this._handleErrorZS||this._handleErrorZS.onHandleError(this._handleErrorDlgt,this._handleErrorCurrZone,E,t)}scheduleTask(E,t){let r=t;if(this._scheduleTaskZS)this._hasTaskZS&&r._zoneDelegates.push(this._hasTaskDlgtOwner),r=this._scheduleTaskZS.onScheduleTask(this._scheduleTaskDlgt,this._scheduleTaskCurrZone,E,t),r||(r=t);else if(t.scheduleFn)t.scheduleFn(t);else{if(t.type!=L)throw new Error("Task is missing scheduleFn.");R(t)}return r}invokeTask(E,t,r,k){return this._invokeTaskZS?this._invokeTaskZS.onInvokeTask(this._invokeTaskDlgt,this._invokeTaskCurrZone,E,t,r,k):t.callback.apply(r,k)}cancelTask(E,t){let r;if(this._cancelTaskZS)r=this._cancelTaskZS.onCancelTask(this._cancelTaskDlgt,this._cancelTaskCurrZone,E,t);else{if(!t.cancelFn)throw Error("Task is not cancelable");r=t.cancelFn(t)}return r}hasTask(E,t){try{this._hasTaskZS&&this._hasTaskZS.onHasTask(this._hasTaskDlgt,this._hasTaskCurrZone,E,t)}catch(r){this.handleError(E,r)}}_updateTaskCount(E,t){const r=this._taskCounts,k=r[E],C=r[E]=k+t;if(C<0)throw new Error("More tasks executed then were scheduled.");0!=k&&0!=C||this.hasTask(this.zone,{microTask:r.microTask>0,macroTask:r.macroTask>0,eventTask:r.eventTask>0,change:E})}}class g{constructor(E,t,r,k,C,J){if(this._zone=null,this.runCount=0,this._zoneDelegates=null,this._state="notScheduled",this.type=E,this.source=t,this.data=k,this.scheduleFn=C,this.cancelFn=J,!r)throw new Error("callback is not defined");this.callback=r;const l=this;this.invoke=E===te&&k&&k.useG?g.invokeTask:function(){return g.invokeTask.call(e,l,this,arguments)}}static invokeTask(E,t,r){E||(E=this),ne++;try{return E.runCount++,E.zone.runTask(E,t,r)}finally{1==ne&&_(),ne--}}get zone(){return this._zone}get state(){return this._state}cancelScheduleRequest(){this._transitionTo(x,z)}_transitionTo(E,t,r){if(this._state!==t&&this._state!==r)throw new Error(`${this.type} '${this.source}': can not transition to '${E}', expecting state '${t}'${r?" or '"+r+"'":""}, was '${this._state}'.`);this._state=E,E==x&&(this._zoneDelegates=null)}toString(){return this.data&&void 0!==this.data.handleId?this.data.handleId.toString():Object.prototype.toString.call(this)}toJSON(){return{type:this.type,state:this.state,source:this.source,zone:this.zone.name,runCount:this.runCount}}}const A=a("setTimeout"),N=a("Promise"),I=a("then");let ee,F=[],H=!1;function q(M){if(ee||e[N]&&(ee=e[N].resolve(0)),ee){let E=ee[I];E||(E=ee.then),E.call(ee,M)}else e[A](M,0)}function R(M){0===ne&&0===F.length&&q(_),M&&F.push(M)}function _(){if(!H){for(H=!0;F.length;){const M=F;F=[];for(let E=0;E<M.length;E++){const t=M[E];try{t.zone.runTask(t,null,null)}catch(r){X.onUnhandledError(r)}}}X.microtaskDrainDone(),H=!1}}const Q={name:"NO ZONE"},x="notScheduled",z="scheduling",j="scheduled",p="running",G="canceling",h="unknown",L="microTask",P="macroTask",te="eventTask",le={},X={symbol:a,currentZoneFrame:()=>U,onUnhandledError:W,microtaskDrainDone:W,scheduleMicroTask:R,showUncaughtError:()=>!d[a("ignoreConsoleErrorUncaughtError")],patchEventTarget:()=>[],patchOnProperties:W,patchMethod:()=>W,bindArguments:()=>[],patchThen:()=>W,patchMacroTask:()=>W,patchEventPrototype:()=>W,isIEOrEdge:()=>!1,getGlobalObjects:()=>{},ObjectDefineProperty:()=>W,ObjectGetOwnPropertyDescriptor:()=>{},ObjectCreate:()=>{},ArraySlice:()=>[],patchClass:()=>W,wrapWithCurrentZone:()=>W,filterProperties:()=>[],attachOriginToPatched:()=>W,_redefineProperty:()=>W,patchCallbacks:()=>W,nativeScheduleMicroTask:q};let U={parent:null,zone:new d(null,null)},ae=null,ne=0;function W(){}o("Zone","Zone"),e.Zone=d}("undefined"!=typeof window&&window||"undefined"!=typeof self&&self||global);const fe=Object.getOwnPropertyDescriptor,ge=Object.defineProperty,me=Object.getPrototypeOf,we=Object.create,Me=Array.prototype.slice,Pe="addEventListener",Oe="removeEventListener",Re=Zone.__symbol__(Pe),Te=Zone.__symbol__(Oe),re="true",oe="false",ye=Zone.__symbol__("");function De(e,n){return Zone.current.wrap(e,n)}function $(e,n,i,o,c){return Zone.current.scheduleMacroTask(e,n,i,o,c)}const O=Zone.__symbol__,se="undefined"!=typeof window,Y=se?window:void 0,K=se&&Y||"object"==typeof self&&self||global;function Ae(e,n){for(let i=e.length-1;i>=0;i--)"function"==typeof e[i]&&(e[i]=De(e[i],n+"_"+i));return e}function Be(e){return!e||!1!==e.writable&&!("function"==typeof e.get&&void 0===e.set)}const Fe="undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope,Ze=!("nw"in K)&&void 0!==K.process&&"[object process]"==={}.toString.call(K.process),je=!Ze&&!Fe&&!(!se||!Y.HTMLElement),Ue=void 0!==K.process&&"[object process]"==={}.toString.call(K.process)&&!Fe&&!(!se||!Y.HTMLElement),Ne={},We=function(e){if(!(e=e||K.event))return;let n=Ne[e.type];n||(n=Ne[e.type]=O("ON_PROPERTY"+e.type));const i=this||e.target||K,o=i[n];let c;if(je&&i===Y&&"error"===e.type){const a=e;c=o&&o.call(this,a.message,a.filename,a.lineno,a.colno,a.error),!0===c&&e.preventDefault()}else c=o&&o.apply(this,arguments),null!=c&&!c&&e.preventDefault();return c};function qe(e,n,i){let o=fe(e,n);if(!o&&i&&fe(i,n)&&(o={enumerable:!0,configurable:!0}),!o||!o.configurable)return;const c=O("on"+n+"patched");if(e.hasOwnProperty(c)&&e[c])return;delete o.writable,delete o.value;const a=o.get,y=o.set,d=n.slice(2);let w=Ne[d];w||(w=Ne[d]=O("ON_PROPERTY"+d)),o.set=function(v){let g=this;!g&&e===K&&(g=K),g&&("function"==typeof g[w]&&g.removeEventListener(d,We),y&&y.call(g,null),g[w]=v,"function"==typeof v&&g.addEventListener(d,We,!1))},o.get=function(){let v=this;if(!v&&e===K&&(v=K),!v)return null;const g=v[w];if(g)return g;if(a){let A=a.call(this);if(A)return o.set.call(this,A),"function"==typeof v.removeAttribute&&v.removeAttribute(n),A}return null},ge(e,n,o),e[c]=!0}function ze(e,n,i){if(n)for(let o=0;o<n.length;o++)qe(e,"on"+n[o],i);else{const o=[];for(const c in e)"on"==c.slice(0,2)&&o.push(c);for(let c=0;c<o.length;c++)qe(e,o[c],i)}}const ce=O("originalInstance");function Ce(e){const n=K[e];if(!n)return;K[O(e)]=n,K[e]=function(){const c=Ae(arguments,e);switch(c.length){case 0:this[ce]=new n;break;case 1:this[ce]=new n(c[0]);break;case 2:this[ce]=new n(c[0],c[1]);break;case 3:this[ce]=new n(c[0],c[1],c[2]);break;case 4:this[ce]=new n(c[0],c[1],c[2],c[3]);break;default:throw new Error("Arg list too long.")}},de(K[e],n);const i=new n(function(){});let o;for(o in i)"XMLHttpRequest"===e&&"responseBlob"===o||function(c){"function"==typeof i[c]?K[e].prototype[c]=function(){return this[ce][c].apply(this[ce],arguments)}:ge(K[e].prototype,c,{set:function(a){"function"==typeof a?(this[ce][c]=De(a,e+"."+c),de(this[ce][c],a)):this[ce][c]=a},get:function(){return this[ce][c]}})}(o);for(o in n)"prototype"!==o&&n.hasOwnProperty(o)&&(K[e][o]=n[o])}function he(e,n,i){let o=e;for(;o&&!o.hasOwnProperty(n);)o=me(o);!o&&e[n]&&(o=e);const c=O(n);let a=null;if(o&&(!(a=o[c])||!o.hasOwnProperty(c))&&(a=o[c]=o[n],Be(o&&fe(o,n)))){const d=i(a,c,n);o[n]=function(){return d(this,arguments)},de(o[n],a)}return a}function lt(e,n,i){let o=null;function c(a){const y=a.data;return y.args[y.cbIdx]=function(){a.invoke.apply(this,arguments)},o.apply(y.target,y.args),a}o=he(e,n,a=>function(y,d){const w=i(y,d);return w.cbIdx>=0&&"function"==typeof d[w.cbIdx]?$(w.name,d[w.cbIdx],w,c):a.apply(y,d)})}function de(e,n){e[O("OriginalDelegate")]=n}let Xe=!1,He=!1;function ft(){if(Xe)return He;Xe=!0;try{const e=Y.navigator.userAgent;(-1!==e.indexOf("MSIE ")||-1!==e.indexOf("Trident/")||-1!==e.indexOf("Edge/"))&&(He=!0)}catch(e){}return He}Zone.__load_patch("ZoneAwarePromise",(e,n,i)=>{const o=Object.getOwnPropertyDescriptor,c=Object.defineProperty,y=i.symbol,d=[],w=!0===e[y("DISABLE_WRAPPING_UNCAUGHT_PROMISE_REJECTION")],v=y("Promise"),g=y("then");i.onUnhandledError=l=>{if(i.showUncaughtError()){const u=l&&l.rejection;u?console.error("Unhandled Promise rejection:",u instanceof Error?u.message:u,"; Zone:",l.zone.name,"; Task:",l.task&&l.task.source,"; Value:",u,u instanceof Error?u.stack:void 0):console.error(l)}},i.microtaskDrainDone=()=>{for(;d.length;){const l=d.shift();try{l.zone.runGuarded(()=>{throw l.throwOriginal?l.rejection:l})}catch(u){I(u)}}};const N=y("unhandledPromiseRejectionHandler");function I(l){i.onUnhandledError(l);try{const u=n[N];"function"==typeof u&&u.call(this,l)}catch(u){}}function F(l){return l&&l.then}function H(l){return l}function ee(l){return t.reject(l)}const q=y("state"),R=y("value"),_=y("finally"),Q=y("parentPromiseValue"),x=y("parentPromiseState"),j=null,p=!0,G=!1;function L(l,u){return s=>{try{X(l,u,s)}catch(f){X(l,!1,f)}}}const P=function(){let l=!1;return function(s){return function(){l||(l=!0,s.apply(null,arguments))}}},le=y("currentTaskTrace");function X(l,u,s){const f=P();if(l===s)throw new TypeError("Promise resolved with itself");if(l[q]===j){let m=null;try{("object"==typeof s||"function"==typeof s)&&(m=s&&s.then)}catch(b){return f(()=>{X(l,!1,b)})(),l}if(u!==G&&s instanceof t&&s.hasOwnProperty(q)&&s.hasOwnProperty(R)&&s[q]!==j)ae(s),X(l,s[q],s[R]);else if(u!==G&&"function"==typeof m)try{m.call(s,f(L(l,u)),f(L(l,!1)))}catch(b){f(()=>{X(l,!1,b)})()}else{l[q]=u;const b=l[R];if(l[R]=s,l[_]===_&&u===p&&(l[q]=l[x],l[R]=l[Q]),u===G&&s instanceof Error){const T=n.currentTask&&n.currentTask.data&&n.currentTask.data.__creationTrace__;T&&c(s,le,{configurable:!0,enumerable:!1,writable:!0,value:T})}for(let T=0;T<b.length;)ne(l,b[T++],b[T++],b[T++],b[T++]);if(0==b.length&&u==G){l[q]=0;let T=s;try{throw new Error("Uncaught (in promise): "+function a(l){return l&&l.toString===Object.prototype.toString?(l.constructor&&l.constructor.name||"")+": "+JSON.stringify(l):l?l.toString():Object.prototype.toString.call(l)}(s)+(s&&s.stack?"\n"+s.stack:""))}catch(S){T=S}w&&(T.throwOriginal=!0),T.rejection=s,T.promise=l,T.zone=n.current,T.task=n.currentTask,d.push(T),i.scheduleMicroTask()}}}return l}const U=y("rejectionHandledHandler");function ae(l){if(0===l[q]){try{const u=n[U];u&&"function"==typeof u&&u.call(this,{rejection:l[R],promise:l})}catch(u){}l[q]=G;for(let u=0;u<d.length;u++)l===d[u].promise&&d.splice(u,1)}}function ne(l,u,s,f,m){ae(l);const b=l[q],T=b?"function"==typeof f?f:H:"function"==typeof m?m:ee;u.scheduleMicroTask("Promise.then",()=>{try{const S=l[R],D=!!s&&_===s[_];D&&(s[Q]=S,s[x]=b);const Z=u.run(T,void 0,D&&T!==ee&&T!==H?[]:[S]);X(s,!0,Z)}catch(S){X(s,!1,S)}},s)}const M=function(){},E=e.AggregateError;class t{static toString(){return"function ZoneAwarePromise() { [native code] }"}static resolve(u){return X(new this(null),p,u)}static reject(u){return X(new this(null),G,u)}static any(u){if(!u||"function"!=typeof u[Symbol.iterator])return Promise.reject(new E([],"All promises were rejected"));const s=[];let f=0;try{for(let T of u)f++,s.push(t.resolve(T))}catch(T){return Promise.reject(new E([],"All promises were rejected"))}if(0===f)return Promise.reject(new E([],"All promises were rejected"));let m=!1;const b=[];return new t((T,S)=>{for(let D=0;D<s.length;D++)s[D].then(Z=>{m||(m=!0,T(Z))},Z=>{b.push(Z),f--,0===f&&(m=!0,S(new E(b,"All promises were rejected")))})})}static race(u){let s,f,m=new this((S,D)=>{s=S,f=D});function b(S){s(S)}function T(S){f(S)}for(let S of u)F(S)||(S=this.resolve(S)),S.then(b,T);return m}static all(u){return t.allWithCallback(u)}static allSettled(u){return(this&&this.prototype instanceof t?this:t).allWithCallback(u,{thenCallback:f=>({status:"fulfilled",value:f}),errorCallback:f=>({status:"rejected",reason:f})})}static allWithCallback(u,s){let f,m,b=new this((Z,V)=>{f=Z,m=V}),T=2,S=0;const D=[];for(let Z of u){F(Z)||(Z=this.resolve(Z));const V=S;try{Z.then(B=>{D[V]=s?s.thenCallback(B):B,T--,0===T&&f(D)},B=>{s?(D[V]=s.errorCallback(B),T--,0===T&&f(D)):m(B)})}catch(B){m(B)}T++,S++}return T-=2,0===T&&f(D),b}constructor(u){const s=this;if(!(s instanceof t))throw new Error("Must be an instanceof Promise.");s[q]=j,s[R]=[];try{const f=P();u&&u(f(L(s,p)),f(L(s,G)))}catch(f){X(s,!1,f)}}get[Symbol.toStringTag](){return"Promise"}get[Symbol.species](){return t}then(u,s){var f;let m=null===(f=this.constructor)||void 0===f?void 0:f[Symbol.species];(!m||"function"!=typeof m)&&(m=this.constructor||t);const b=new m(M),T=n.current;return this[q]==j?this[R].push(T,b,u,s):ne(this,T,b,u,s),b}catch(u){return this.then(null,u)}finally(u){var s;let f=null===(s=this.constructor)||void 0===s?void 0:s[Symbol.species];(!f||"function"!=typeof f)&&(f=t);const m=new f(M);m[_]=_;const b=n.current;return this[q]==j?this[R].push(b,m,u,u):ne(this,b,m,u,u),m}}t.resolve=t.resolve,t.reject=t.reject,t.race=t.race,t.all=t.all;const r=e[v]=e.Promise;e.Promise=t;const k=y("thenPatched");function C(l){const u=l.prototype,s=o(u,"then");if(s&&(!1===s.writable||!s.configurable))return;const f=u.then;u[g]=f,l.prototype.then=function(m,b){return new t((S,D)=>{f.call(this,S,D)}).then(m,b)},l[k]=!0}return i.patchThen=C,r&&(C(r),he(e,"fetch",l=>function J(l){return function(u,s){let f=l.apply(u,s);if(f instanceof t)return f;let m=f.constructor;return m[k]||C(m),f}}(l))),Promise[n.__symbol__("uncaughtPromiseErrors")]=d,t}),Zone.__load_patch("toString",e=>{const n=Function.prototype.toString,i=O("OriginalDelegate"),o=O("Promise"),c=O("Error"),a=function(){if("function"==typeof this){const v=this[i];if(v)return"function"==typeof v?n.call(v):Object.prototype.toString.call(v);if(this===Promise){const g=e[o];if(g)return n.call(g)}if(this===Error){const g=e[c];if(g)return n.call(g)}}return n.call(this)};a[i]=n,Function.prototype.toString=a;const y=Object.prototype.toString;Object.prototype.toString=function(){return"function"==typeof Promise&&this instanceof Promise?"[object Promise]":y.call(this)}});let ke=!1;if("undefined"!=typeof window)try{const e=Object.defineProperty({},"passive",{get:function(){ke=!0}});window.addEventListener("test",e,e),window.removeEventListener("test",e,e)}catch(e){ke=!1}const ht={useG:!0},ie={},Ye={},$e=new RegExp("^"+ye+"(\\w+)(true|false)$"),Ke=O("propagationStopped");function Je(e,n){const i=(n?n(e):e)+oe,o=(n?n(e):e)+re,c=ye+i,a=ye+o;ie[e]={},ie[e][oe]=c,ie[e][re]=a}function dt(e,n,i,o){const c=o&&o.add||Pe,a=o&&o.rm||Oe,y=o&&o.listeners||"eventListeners",d=o&&o.rmAll||"removeAllListeners",w=O(c),v="."+c+":",N=function(R,_,Q){if(R.isRemoved)return;const x=R.callback;let z;"object"==typeof x&&x.handleEvent&&(R.callback=p=>x.handleEvent(p),R.originalDelegate=x);try{R.invoke(R,_,[Q])}catch(p){z=p}const j=R.options;return j&&"object"==typeof j&&j.once&&_[a].call(_,Q.type,R.originalDelegate?R.originalDelegate:R.callback,j),z};function I(R,_,Q){if(!(_=_||e.event))return;const x=R||_.target||e,z=x[ie[_.type][Q?re:oe]];if(z){const j=[];if(1===z.length){const p=N(z[0],x,_);p&&j.push(p)}else{const p=z.slice();for(let G=0;G<p.length&&(!_||!0!==_[Ke]);G++){const h=N(p[G],x,_);h&&j.push(h)}}if(1===j.length)throw j[0];for(let p=0;p<j.length;p++){const G=j[p];n.nativeScheduleMicroTask(()=>{throw G})}}}const F=function(R){return I(this,R,!1)},H=function(R){return I(this,R,!0)};function ee(R,_){if(!R)return!1;let Q=!0;_&&void 0!==_.useG&&(Q=_.useG);const x=_&&_.vh;let z=!0;_&&void 0!==_.chkDup&&(z=_.chkDup);let j=!1;_&&void 0!==_.rt&&(j=_.rt);let p=R;for(;p&&!p.hasOwnProperty(c);)p=me(p);if(!p&&R[c]&&(p=R),!p||p[w])return!1;const G=_&&_.eventNameToString,h={},L=p[w]=p[c],P=p[O(a)]=p[a],te=p[O(y)]=p[y],le=p[O(d)]=p[d];let X;function U(s,f){return!ke&&"object"==typeof s&&s?!!s.capture:ke&&f?"boolean"==typeof s?{capture:s,passive:!0}:s?"object"==typeof s&&!1!==s.passive?Object.assign(Object.assign({},s),{passive:!0}):s:{passive:!0}:s}_&&_.prepend&&(X=p[O(_.prepend)]=p[_.prepend]);const t=Q?function(s){if(!h.isExisting)return L.call(h.target,h.eventName,h.capture?H:F,h.options)}:function(s){return L.call(h.target,h.eventName,s.invoke,h.options)},r=Q?function(s){if(!s.isRemoved){const f=ie[s.eventName];let m;f&&(m=f[s.capture?re:oe]);const b=m&&s.target[m];if(b)for(let T=0;T<b.length;T++)if(b[T]===s){b.splice(T,1),s.isRemoved=!0,0===b.length&&(s.allRemoved=!0,s.target[m]=null);break}}if(s.allRemoved)return P.call(s.target,s.eventName,s.capture?H:F,s.options)}:function(s){return P.call(s.target,s.eventName,s.invoke,s.options)},C=_&&_.diff?_.diff:function(s,f){const m=typeof f;return"function"===m&&s.callback===f||"object"===m&&s.originalDelegate===f},J=Zone[O("UNPATCHED_EVENTS")],l=e[O("PASSIVE_EVENTS")],u=function(s,f,m,b,T=!1,S=!1){return function(){const D=this||e;let Z=arguments[0];_&&_.transferEventName&&(Z=_.transferEventName(Z));let V=arguments[1];if(!V)return s.apply(this,arguments);if(Ze&&"uncaughtException"===Z)return s.apply(this,arguments);let B=!1;if("function"!=typeof V){if(!V.handleEvent)return s.apply(this,arguments);B=!0}if(x&&!x(s,V,D,arguments))return;const _e=ke&&!!l&&-1!==l.indexOf(Z),ue=U(arguments[2],_e);if(J)for(let pe=0;pe<J.length;pe++)if(Z===J[pe])return _e?s.call(D,Z,V,ue):s.apply(this,arguments);const Ge=!!ue&&("boolean"==typeof ue||ue.capture),nt=!(!ue||"object"!=typeof ue)&&ue.once,mt=Zone.current;let Ve=ie[Z];Ve||(Je(Z,G),Ve=ie[Z]);const rt=Ve[Ge?re:oe];let Le,be=D[rt],ot=!1;if(be){if(ot=!0,z)for(let pe=0;pe<be.length;pe++)if(C(be[pe],V))return}else be=D[rt]=[];const st=D.constructor.name,it=Ye[st];it&&(Le=it[Z]),Le||(Le=st+f+(G?G(Z):Z)),h.options=ue,nt&&(h.options.once=!1),h.target=D,h.capture=Ge,h.eventName=Z,h.isExisting=ot;const Se=Q?ht:void 0;Se&&(Se.taskData=h);const Ee=mt.scheduleEventTask(Le,V,Se,m,b);return h.target=null,Se&&(Se.taskData=null),nt&&(ue.once=!0),!ke&&"boolean"==typeof Ee.options||(Ee.options=ue),Ee.target=D,Ee.capture=Ge,Ee.eventName=Z,B&&(Ee.originalDelegate=V),S?be.unshift(Ee):be.push(Ee),T?D:void 0}};return p[c]=u(L,v,t,r,j),X&&(p.prependListener=u(X,".prependListener:",function(s){return X.call(h.target,h.eventName,s.invoke,h.options)},r,j,!0)),p[a]=function(){const s=this||e;let f=arguments[0];_&&_.transferEventName&&(f=_.transferEventName(f));const m=arguments[2],b=!!m&&("boolean"==typeof m||m.capture),T=arguments[1];if(!T)return P.apply(this,arguments);if(x&&!x(P,T,s,arguments))return;const S=ie[f];let D;S&&(D=S[b?re:oe]);const Z=D&&s[D];if(Z)for(let V=0;V<Z.length;V++){const B=Z[V];if(C(B,T))return Z.splice(V,1),B.isRemoved=!0,0===Z.length&&(B.allRemoved=!0,s[D]=null,"string"==typeof f)&&(s[ye+"ON_PROPERTY"+f]=null),B.zone.cancelTask(B),j?s:void 0}return P.apply(this,arguments)},p[y]=function(){const s=this||e;let f=arguments[0];_&&_.transferEventName&&(f=_.transferEventName(f));const m=[],b=Qe(s,G?G(f):f);for(let T=0;T<b.length;T++){const S=b[T];m.push(S.originalDelegate?S.originalDelegate:S.callback)}return m},p[d]=function(){const s=this||e;let f=arguments[0];if(f){_&&_.transferEventName&&(f=_.transferEventName(f));const m=ie[f];if(m){const S=s[m[oe]],D=s[m[re]];if(S){const Z=S.slice();for(let V=0;V<Z.length;V++){const B=Z[V];this[a].call(this,f,B.originalDelegate?B.originalDelegate:B.callback,B.options)}}if(D){const Z=D.slice();for(let V=0;V<Z.length;V++){const B=Z[V];this[a].call(this,f,B.originalDelegate?B.originalDelegate:B.callback,B.options)}}}}else{const m=Object.keys(s);for(let b=0;b<m.length;b++){const S=$e.exec(m[b]);let D=S&&S[1];D&&"removeListener"!==D&&this[d].call(this,D)}this[d].call(this,"removeListener")}if(j)return this},de(p[c],L),de(p[a],P),le&&de(p[d],le),te&&de(p[y],te),!0}let q=[];for(let R=0;R<i.length;R++)q[R]=ee(i[R],o);return q}function Qe(e,n){if(!n){const a=[];for(let y in e){const d=$e.exec(y);let w=d&&d[1];if(w&&(!n||w===n)){const v=e[y];if(v)for(let g=0;g<v.length;g++)a.push(v[g])}}return a}let i=ie[n];i||(Je(n),i=ie[n]);const o=e[i[oe]],c=e[i[re]];return o?c?o.concat(c):o.slice():c?c.slice():[]}function _t(e,n){const i=e.Event;i&&i.prototype&&n.patchMethod(i.prototype,"stopImmediatePropagation",o=>function(c,a){c[Ke]=!0,o&&o.apply(c,a)})}function Et(e,n,i,o,c){const a=Zone.__symbol__(o);if(n[a])return;const y=n[a]=n[o];n[o]=function(d,w,v){return w&&w.prototype&&c.forEach(function(g){const A=`${i}.${o}::`+g,N=w.prototype;try{if(N.hasOwnProperty(g)){const I=e.ObjectGetOwnPropertyDescriptor(N,g);I&&I.value?(I.value=e.wrapWithCurrentZone(I.value,A),e._redefineProperty(w.prototype,g,I)):N[g]&&(N[g]=e.wrapWithCurrentZone(N[g],A))}else N[g]&&(N[g]=e.wrapWithCurrentZone(N[g],A))}catch(I){}}),y.call(n,d,w,v)},e.attachOriginToPatched(n[o],y)}function et(e,n,i){if(!i||0===i.length)return n;const o=i.filter(a=>a.target===e);if(!o||0===o.length)return n;const c=o[0].ignoreProperties;return n.filter(a=>-1===c.indexOf(a))}function tt(e,n,i,o){e&&ze(e,et(e,n,i),o)}function xe(e){return Object.getOwnPropertyNames(e).filter(n=>n.startsWith("on")&&n.length>2).map(n=>n.substring(2))}Zone.__load_patch("util",(e,n,i)=>{const o=xe(e);i.patchOnProperties=ze,i.patchMethod=he,i.bindArguments=Ae,i.patchMacroTask=lt;const c=n.__symbol__("BLACK_LISTED_EVENTS"),a=n.__symbol__("UNPATCHED_EVENTS");e[a]&&(e[c]=e[a]),e[c]&&(n[c]=n[a]=e[c]),i.patchEventPrototype=_t,i.patchEventTarget=dt,i.isIEOrEdge=ft,i.ObjectDefineProperty=ge,i.ObjectGetOwnPropertyDescriptor=fe,i.ObjectCreate=we,i.ArraySlice=Me,i.patchClass=Ce,i.wrapWithCurrentZone=De,i.filterProperties=et,i.attachOriginToPatched=de,i._redefineProperty=Object.defineProperty,i.patchCallbacks=Et,i.getGlobalObjects=()=>({globalSources:Ye,zoneSymbolEventNames:ie,eventNames:o,isBrowser:je,isMix:Ue,isNode:Ze,TRUE_STR:re,FALSE_STR:oe,ZONE_SYMBOL_PREFIX:ye,ADD_EVENT_LISTENER_STR:Pe,REMOVE_EVENT_LISTENER_STR:Oe})});const Ie=O("zoneTask");function ve(e,n,i,o){let c=null,a=null;i+=o;const y={};function d(v){const g=v.data;return g.args[0]=function(){return v.invoke.apply(this,arguments)},g.handleId=c.apply(e,g.args),v}function w(v){return a.call(e,v.data.handleId)}c=he(e,n+=o,v=>function(g,A){if("function"==typeof A[0]){const N={isPeriodic:"Interval"===o,delay:"Timeout"===o||"Interval"===o?A[1]||0:void 0,args:A},I=A[0];A[0]=function(){try{return I.apply(this,arguments)}finally{N.isPeriodic||("number"==typeof N.handleId?delete y[N.handleId]:N.handleId&&(N.handleId[Ie]=null))}};const F=$(n,A[0],N,d,w);if(!F)return F;const H=F.data.handleId;return"number"==typeof H?y[H]=F:H&&(H[Ie]=F),H&&H.ref&&H.unref&&"function"==typeof H.ref&&"function"==typeof H.unref&&(F.ref=H.ref.bind(H),F.unref=H.unref.bind(H)),"number"==typeof H||H?H:F}return v.apply(e,A)}),a=he(e,i,v=>function(g,A){const N=A[0];let I;"number"==typeof N?I=y[N]:(I=N&&N[Ie],I||(I=N)),I&&"string"==typeof I.type?"notScheduled"!==I.state&&(I.cancelFn&&I.data.isPeriodic||0===I.runCount)&&("number"==typeof N?delete y[N]:N&&(N[Ie]=null),I.zone.cancelTask(I)):v.apply(e,A)})}Zone.__load_patch("legacy",e=>{const n=e[Zone.__symbol__("legacyPatch")];n&&n()}),Zone.__load_patch("queueMicrotask",(e,n,i)=>{i.patchMethod(e,"queueMicrotask",o=>function(c,a){n.current.scheduleMicroTask("queueMicrotask",a[0])})}),Zone.__load_patch("timers",e=>{const n="set",i="clear";ve(e,n,i,"Timeout"),ve(e,n,i,"Interval"),ve(e,n,i,"Immediate")}),Zone.__load_patch("requestAnimationFrame",e=>{ve(e,"request","cancel","AnimationFrame"),ve(e,"mozRequest","mozCancel","AnimationFrame"),ve(e,"webkitRequest","webkitCancel","AnimationFrame")}),Zone.__load_patch("blocking",(e,n)=>{const i=["alert","prompt","confirm"];for(let o=0;o<i.length;o++)he(e,i[o],(a,y,d)=>function(w,v){return n.current.run(a,e,v,d)})}),Zone.__load_patch("EventTarget",(e,n,i)=>{(function gt(e,n){n.patchEventPrototype(e,n)})(e,i),function pt(e,n){if(Zone[n.symbol("patchEventTarget")])return;const{eventNames:i,zoneSymbolEventNames:o,TRUE_STR:c,FALSE_STR:a,ZONE_SYMBOL_PREFIX:y}=n.getGlobalObjects();for(let w=0;w<i.length;w++){const v=i[w],N=y+(v+a),I=y+(v+c);o[v]={},o[v][a]=N,o[v][c]=I}const d=e.EventTarget;d&&d.prototype&&n.patchEventTarget(e,n,[d&&d.prototype])}(e,i);const o=e.XMLHttpRequestEventTarget;o&&o.prototype&&i.patchEventTarget(e,i,[o.prototype])}),Zone.__load_patch("MutationObserver",(e,n,i)=>{Ce("MutationObserver"),Ce("WebKitMutationObserver")}),Zone.__load_patch("IntersectionObserver",(e,n,i)=>{Ce("IntersectionObserver")}),Zone.__load_patch("FileReader",(e,n,i)=>{Ce("FileReader")}),Zone.__load_patch("on_property",(e,n,i)=>{!function Tt(e,n){if(Ze&&!Ue||Zone[e.symbol("patchEvents")])return;const i=n.__Zone_ignore_on_properties;let o=[];if(je){const c=window;o=o.concat(["Document","SVGElement","Element","HTMLElement","HTMLBodyElement","HTMLMediaElement","HTMLFrameSetElement","HTMLFrameElement","HTMLIFrameElement","HTMLMarqueeElement","Worker"]);const a=function ut(){try{const e=Y.navigator.userAgent;if(-1!==e.indexOf("MSIE ")||-1!==e.indexOf("Trident/"))return!0}catch(e){}return!1}()?[{target:c,ignoreProperties:["error"]}]:[];tt(c,xe(c),i&&i.concat(a),me(c))}o=o.concat(["XMLHttpRequest","XMLHttpRequestEventTarget","IDBIndex","IDBRequest","IDBOpenDBRequest","IDBDatabase","IDBTransaction","IDBCursor","WebSocket"]);for(let c=0;c<o.length;c++){const a=n[o[c]];a&&a.prototype&&tt(a.prototype,xe(a.prototype),i)}}(i,e)}),Zone.__load_patch("customElements",(e,n,i)=>{!function yt(e,n){const{isBrowser:i,isMix:o}=n.getGlobalObjects();(i||o)&&e.customElements&&"customElements"in e&&n.patchCallbacks(n,e.customElements,"customElements","define",["connectedCallback","disconnectedCallback","adoptedCallback","attributeChangedCallback"])}(e,i)}),Zone.__load_patch("XHR",(e,n)=>{!function w(v){const g=v.XMLHttpRequest;if(!g)return;const A=g.prototype;let I=A[Re],F=A[Te];if(!I){const h=v.XMLHttpRequestEventTarget;if(h){const L=h.prototype;I=L[Re],F=L[Te]}}const H="readystatechange",ee="scheduled";function q(h){const L=h.data,P=L.target;P[a]=!1,P[d]=!1;const te=P[c];I||(I=P[Re],F=P[Te]),te&&F.call(P,H,te);const le=P[c]=()=>{if(P.readyState===P.DONE)if(!L.aborted&&P[a]&&h.state===ee){const U=P[n.__symbol__("loadfalse")];if(0!==P.status&&U&&U.length>0){const ae=h.invoke;h.invoke=function(){const ne=P[n.__symbol__("loadfalse")];for(let W=0;W<ne.length;W++)ne[W]===h&&ne.splice(W,1);!L.aborted&&h.state===ee&&ae.call(h)},U.push(h)}else h.invoke()}else!L.aborted&&!1===P[a]&&(P[d]=!0)};return I.call(P,H,le),P[i]||(P[i]=h),p.apply(P,L.args),P[a]=!0,h}function R(){}function _(h){const L=h.data;return L.aborted=!0,G.apply(L.target,L.args)}const Q=he(A,"open",()=>function(h,L){return h[o]=0==L[2],h[y]=L[1],Q.apply(h,L)}),z=O("fetchTaskAborting"),j=O("fetchTaskScheduling"),p=he(A,"send",()=>function(h,L){if(!0===n.current[j]||h[o])return p.apply(h,L);{const P={target:h,url:h[y],isPeriodic:!1,args:L,aborted:!1},te=$("XMLHttpRequest.send",R,P,q,_);h&&!0===h[d]&&!P.aborted&&te.state===ee&&te.invoke()}}),G=he(A,"abort",()=>function(h,L){const P=function N(h){return h[i]}(h);if(P&&"string"==typeof P.type){if(null==P.cancelFn||P.data&&P.data.aborted)return;P.zone.cancelTask(P)}else if(!0===n.current[z])return G.apply(h,L)})}(e);const i=O("xhrTask"),o=O("xhrSync"),c=O("xhrListener"),a=O("xhrScheduled"),y=O("xhrURL"),d=O("xhrErrorBeforeScheduled")}),Zone.__load_patch("geolocation",e=>{e.navigator&&e.navigator.geolocation&&function at(e,n){const i=e.constructor.name;for(let o=0;o<n.length;o++){const c=n[o],a=e[c];if(a){if(!Be(fe(e,c)))continue;e[c]=(d=>{const w=function(){return d.apply(this,Ae(arguments,i+"."+c))};return de(w,d),w})(a)}}}(e.navigator.geolocation,["getCurrentPosition","watchPosition"])}),Zone.__load_patch("PromiseRejectionEvent",(e,n)=>{function i(o){return function(c){Qe(e,o).forEach(y=>{const d=e.PromiseRejectionEvent;if(d){const w=new d(o,{promise:c.promise,reason:c.rejection});y.invoke(w)}})}}e.PromiseRejectionEvent&&(n[O("unhandledPromiseRejectionHandler")]=i("unhandledrejection"),n[O("rejectionHandledHandler")]=i("rejectionhandled"))})}},fe=>{fe(fe.s=955)}]);