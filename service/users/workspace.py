import uuid, json, time
import uuid, json, time
from typing import List, Optional, Generator, Dict, Any
from models.base import model_factory
from repository.users.workspace_chat import (
    get_chat_history as core_get_chat_history,
    save_chat as core_save_chat,
    get_workspace_chat_history as core_get_workspace_chat_history,
)
from repository.users.workspace_main import (
    get_all_workspaces as core_get_all_workspaces,
    create_workspace as core_create_workspace,
    get_workspace_by_slug as core_get_workspace_by_slug,
    delete_workspace as core_delete_workspace,
    update_workspace as core_update_workspace,
    get_workspace_id_from_slug as core_get_workspace_id_from_slug
)
from errors import *
from config import config


class WorkspaceService:
    """워크스페이스 관련 비즈니스 로직을 처리하는 서비스 클래스"""
    
    @staticmethod
    def get_all_workspaces() -> List[Dict[str, Any]]:
        """모든 워크스페이스 목록 조회"""
        try:
            return core_get_all_workspaces()
        except Exception as e:
            raise InternalServerError(f"워크스페이스 목록 조회 중 오류 발생: {str(e)}")
    
    @staticmethod
    def create_workspace(
        name: str,
        similarity_threshold: float,
        open_ai_temp: float,
        open_ai_history: int,
        system_prompt: Optional[str] = None,
        query_refusal_response: Optional[str] = None,
        chat_mode: str = "chat",
        top_n: int = 4
    ) -> Dict[str, Any]:
        """새 워크스페이스 생성"""
        try:
            workspace = core_create_workspace(
                name=name,
                similarity_threshold=similarity_threshold,
                open_ai_temp=open_ai_temp,
                open_ai_history=open_ai_history,
                system_prompt=system_prompt or "",
                query_refusal_response=query_refusal_response or "",
                chat_mode=chat_mode,
                top_n=top_n,
            )
            
            return {
                "id": workspace["id"],
                "name": workspace["name"],
                "slug": workspace["slug"],
                "createdAt": workspace["createdAt"],
                "lastUpdatedAt": workspace["lastUpdatedAt"],
                "openAiPrompt": workspace["openAiPrompt"],
                "openAiHistory": workspace["openAiHistory"],
                "openAiTemp": workspace["openAiTemp"]
            }
        except ValueError as e:
            if "이미 사용중인 slug" in str(e):
                raise WorkspaceAlreadyExistsError(str(e).split(":")[1].strip())
            raise BadRequestError(str(e))
        except Exception as e:
            raise InternalServerError(f"워크스페이스 생성 중 오류 발생: {str(e)}")
    
    @staticmethod
    def get_workspace_by_slug(slug: str) -> Dict[str, Any]:
        """워크스페이스 정보 조회"""
        try:
            workspace = core_get_workspace_by_slug(slug)
            if not workspace:
                raise WorkspaceNotFoundError(slug)
            return workspace
        except WorkspaceNotFoundError:
            raise
        except Exception as e:
            raise InternalServerError(f"워크스페이스 조회 중 오류 발생: {str(e)}")
    
    @staticmethod
    def delete_workspace(slug: str) -> Dict[str, Any]:
        """워크스페이스 삭제"""
        try:
            core_delete_workspace(slug)
            return {
                "success": True,
                "message": f"워크스페이스 '{slug}'가 성공적으로 삭제완료되었습니다."
            }
        except ValueError as e:
            raise WorkspaceNotFoundError(slug)
        except Exception as e:
            raise DatabaseError(f"워크스페이스 삭제 중 오류 발생: {str(e)}")    
    @staticmethod
    def get_workspace_chat_history(
        slug: str, 
        api_session_id: Optional[str] = None, 
        limit: int = 20, 
        order_by: str = "createdAt"
    ) -> List[Dict[str, Any]]:
        """워크스페이스 채팅 히스토리 조회"""
        try:
            if limit <= 0 or limit > 100:
                raise BadRequestError("limit는 1~100 사이의 값이어야 합니다.")
            
            history = core_get_workspace_chat_history(slug, api_session_id, limit, order_by)
            return history
        except BadRequestError:
            raise
        except Exception as e:
            raise InternalServerError(f"채팅 히스토리 조회 중 오류 발생: {str(e)}")
    
    @staticmethod
    def update_workspace(slug: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """워크스페이스 설정 업데이트"""
        try:
            # slug 검증
            if not slug or slug.strip() == "":
                raise BadRequestError("유효한 워크스페이스 slug가 필요합니다")
            
            # 업데이트할 데이터 검증
            if not update_data:
                raise BadRequestError("업데이트할 데이터가 없습니다")
            
            # 데이터 유효성 검증
            if "openAiTemp" in update_data and update_data["openAiTemp"] is not None:
                if not (0.0 <= update_data["openAiTemp"] <= 2.0):
                    raise BadRequestError("openAiTemp는 0.0과 2.0 사이의 값이어야 합니다")
            
            if "openAiHistory" in update_data and update_data["openAiHistory"] is not None:
                if not (1 <= update_data["openAiHistory"] <= 1000):
                    raise BadRequestError("openAiHistory는 1과 1000 사이의 값이어야 합니다")
            
            if "name" in update_data and update_data["name"] is not None:
                if not update_data["name"].strip():
                    raise BadRequestError("워크스페이스 이름은 비어있을 수 없습니다")
            
            # 워크스페이스 업데이트
            workspace = core_update_workspace(slug, update_data)
            return workspace
            
        except ValueError as e:
            if "찾을 수 없습니다" in str(e):
                raise NotFoundError(str(e))
            raise BadRequestError(str(e))
        except BadRequestError:
            raise
        except NotFoundError:
            raise
        except Exception as e:
            raise InternalServerError(f"워크스페이스 업데이트 중 오류 발생: {str(e)}")    
    @staticmethod
    def stream_chat(
        slug: str, 
        model: str, 
        message: str, 
        session_id: Optional[str] = "", 
        attachments: Optional[List] = None,
        mode: str = "chat",
        reset: bool = False
    ) -> Generator[str, None, None]:
        """스트림 채팅 처리"""
        try:
            # 1) 이전 대화 히스토리 + 이번 질문 messages 생성
            messages = core_get_chat_history(slug, session_id or "")
            messages.append({"role": "user", "content": message})

            # 2) 모델 인스턴스 생성
            chat_model = model_factory(model)
            generator = chat_model.stream_chat(messages)

            chat_id = str(uuid.uuid4())
            sources = []
            error_msg = None

            # 3) 스트리밍 응답 제너레이터
            full_response = []
            start_time = time.time()
            token_count = 0
            
            try:
                for chunk in generator:
                    if not chunk or chunk.strip() == "":  # 빈 청크는 스킵
                        continue

                    full_response.append(chunk)
                    token_count += 1

                    yield json.dumps({
                        "id": chat_id,
                        "type": "textResponse",
                        "textResponse": chunk,
                        "sources": sources,
                        "close": False,
                        "error": error_msg
                    }, ensure_ascii=False) + "\n"
                    
            finally:
                duration = time.time() - start_time
                # 마지막 패킷(close=True)
                yield json.dumps({
                    "id": chat_id,
                    "type": "textResponse",
                    "textResponse": "",
                    "sources": sources,
                    "close": True,
                    "error": error_msg
                }, ensure_ascii=False) + "\n"

                # 4) DB 저장 (오류가 나도 스트림은 유지)
                try:
                    response_text = "".join(full_response)
                    metrics = {
                        "completion_tokens": len(response_text.split()),
                        "prompt_tokens": len(message.split()),
                        "total_tokens": len(response_text.split()) + len(message.split()),
                        "model": model,
                        "outputTps": token_count / duration if duration > 0 else 0,
                        "duration": round(duration, 3)
                    }
                    core_save_chat(
                        workspace_id=core_get_workspace_id_from_slug(slug),
                        prompt=message,
                        response="".join(full_response),
                        user_id=0,  # 실제 사용자 ID로 변경
                        session_id=session_id or "",
                        sources=sources,
                        attachments=attachments or [],
                        metrics=metrics
                    )
                except Exception as e:
                    print(f"채팅 저장 오류: {e}")  # 로깅으로 대체 가능

        except Exception as e:
            # 에러 발생 시 에러 응답
            yield json.dumps({
                "id": None,
                "type": "abort",
                "textResponse": "",
                "sources": [],
                "close": True,
                "error": str(e)
            }, ensure_ascii=False) + "\n"    
    @staticmethod
    def get_chat_history(slug: str, session_id: str = "") -> List[Dict[str, Any]]:
        """채팅 히스토리 조회"""
        return core_get_chat_history(slug, session_id)
    
    @staticmethod
    def get_workspace_id_from_slug(slug: str) -> int:
        """slug로부터 워크스페이스 ID 조회"""
        return core_get_workspace_id_from_slug(slug)
    
    @staticmethod
    def save_chat(
        workspace_id: int,
        prompt: str,
        response: str,
        user_id: int,
        session_id: str,
        sources: List,
        attachments: List,
        metrics: Dict[str, Any]
    ):
        """채팅 저장"""
        return core_save_chat(
            workspace_id=workspace_id,
            prompt=prompt,
            response=response,
            user_id=user_id,
            session_id=session_id,
            sources=sources,
            attachments=attachments,
            metrics=metrics
        )